"""Unshielded, obstacle-free reference rollouts for every task in the task set
(27 tasks: everything that can start from a fresh initial state, minus
stack/unstack). Their seeds are disjoint from every evaluation seed.

Per task, rebuilt from all checkpoints when the task finishes:
- rollouts_<task>.npz: raw per-step arm frames, rollout ids, reached flags
  (used for the door contact points, scripts/build_door_contact_offsets.py);
- workspace_<task>.npz: the point cloud scene obstacles are drawn from
  (shortstop.calvin.rollout_collection).

Checkpointed: each rollout is saved on its own and a re-run skips the ones
on disk; a manifest refuses to resume under different settings. Set
TASK_NAMES to run a subset.

Run from WSL2, inside `mdt_env`:

    cd SHORTSTOP
    python scripts/collect_calvin_unshielded_rollouts.py
"""
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import calvin_env  # noqa: F401 -- import side effects only, matches every other CALVIN script
import hydra
import numpy as np
from pytorch_lightning import seed_everything

REPO_ROOT = Path(__file__).resolve().parents[1]
MDT_POLICY_ROOT = REPO_ROOT / "mdt_policy"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(MDT_POLICY_ROOT))

from shortstop.calvin.baseline_runner import make_logger, setup_env_and_policy  # noqa: E402
from shortstop.calvin.config import REPLAN_STEPS  # noqa: E402
from shortstop.calvin.experiment import run_calvin_unshielded_sequence  # noqa: E402
from shortstop.calvin.independent_tasks import _enumerate_initial_states  # noqa: E402
from shortstop.calvin.rollout_collection import (  # noqa: E402
    check_or_write_manifest, eligible_task_names, load_rollout_checkpoint,
    raw_rollout_arrays, rollout_checkpoint_path, save_rollout_checkpoint, trials_for_task, workspace_arrays,
)

# None = all 27 eligible tasks (see module docstring); or a list of names.
TASK_NAMES = None

# Rollouts per task type. Safe to raise later: existing checkpoints are
# reused and only the new rollout ids run.
N_ROLLOUTS = 50

# Fixed (not timestamped) so every re-run finds the previous checkpoints.
# Use a new directory for a deliberately different collection.
CHECKPOINT_DIR = REPO_ROOT / "outputs" / "unshielded_rollouts"

# Workspace: interior samples per link segment per step (0 = frames only),
# and whether failed rollouts are pooled too. Both only affect the derived
# files, not the checkpoints, so they can change between runs.
LINK_SAMPLES_PER_SEGMENT = 3
WORKSPACE_ONLY_SUCCESSFUL = True

TRIALS_RNG_SEED = 0
ROLLOUT_SEED_BASE = 2_000_000


def _write_derived_files(task_name, records, ep_len, log):
    """Rebuilds this task's output files from every checkpointed rollout."""
    n_success = sum(r["reached"] for r in records)
    meta = dict(
        task_name=task_name, n_rollouts=len(records), n_success=n_success, ep_len=ep_len,
        trials_rng_seed=TRIALS_RNG_SEED, rollout_seed_base=ROLLOUT_SEED_BASE,
    )
    np.savez(CHECKPOINT_DIR / f"rollouts_{task_name}.npz", **raw_rollout_arrays(records), **meta)

    workspace = workspace_arrays(records, LINK_SAMPLES_PER_SEGMENT, WORKSPACE_ONLY_SUCCESSFUL)
    if workspace is not None:
        np.savez(
            CHECKPOINT_DIR / f"workspace_{task_name}.npz", **workspace, **meta,
            only_successful=WORKSPACE_ONLY_SUCCESSFUL, link_samples_per_segment=LINK_SAMPLES_PER_SEGMENT,
        )

    n_points = 0 if workspace is None else len(workspace["points"])
    log(f"[collect] {task_name} DONE n_success={n_success}/{len(records)} workspace_points={n_points}")
    return n_success


@hydra.main(config_path="../mdt_policy/conf", config_name="mdt_evaluate")
def main(cfg):
    from mdt.evaluation.multistep_sequences import tasks, valid_task
    from mdt.evaluation.utils import get_env_state_for_initial_condition

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    # Append, one log per invocation, all in the same checkpoint dir.
    log = make_logger(CHECKPOINT_DIR / f"run_{datetime.now():%Y%m%d_%H%M%S}.log")

    # Everything that changes what a single rollout IS. Not N_ROLLOUTS,
    # TASK_NAMES or the derived-file settings -- those are safe to change.
    check_or_write_manifest(str(CHECKPOINT_DIR), {
        "ep_len": int(cfg.ep_len),
        "replan_steps": int(REPLAN_STEPS),
        "trials_rng_seed": TRIALS_RNG_SEED,
        "rollout_seed_base": ROLLOUT_SEED_BASE,
        "train_folder": str(cfg.train_folder),
        "dataset_path": str(cfg.dataset_path),
    })

    initial_states = _enumerate_initial_states()
    eligible, chain_only = eligible_task_names(tasks, initial_states, valid_task)
    task_names = eligible if TASK_NAMES is None else list(TASK_NAMES)
    unknown = [name for name in task_names if name not in eligible]
    if unknown:
        raise ValueError(f"not collectable as single-task trials: {unknown} (eligible: {eligible})")

    pending = {
        name: [i for i in range(N_ROLLOUTS)
               if not os.path.exists(rollout_checkpoint_path(CHECKPOINT_DIR, name, i))]
        for name in task_names
    }
    n_pending = sum(len(ids) for ids in pending.values())
    log(f"[collect] {len(task_names)} tasks x {N_ROLLOUTS} rollouts; "
        f"{len(task_names) * N_ROLLOUTS - n_pending} already checkpointed, {n_pending} to run")
    log(f"[collect] skipped chain-only tasks (no valid fresh start): {chain_only}")
    log(f"[collect] ep_len={cfg.ep_len} checkpoint_dir={CHECKPOINT_DIR}")

    env = policy = task_oracle = val_annotations = lang_embeddings = None
    if n_pending:
        seed_everything(0, workers=True)
        # n_candidates=1: plain unshielded rollouts, nothing to rank.
        env, policy, task_oracle, val_annotations, lang_embeddings = setup_env_and_policy(cfg, n_candidates=1)

    started = time.perf_counter()
    n_done = 0
    summary = []
    for task_name in task_names:
        trials = trials_for_task(task_name, tasks, initial_states, valid_task, N_ROLLOUTS, TRIALS_RNG_SEED)
        for rollout_id in pending[task_name]:
            initial_state, eval_sequence = trials[rollout_id]
            seed_everything(ROLLOUT_SEED_BASE + rollout_id, workers=True)
            t0 = time.perf_counter()
            attempts = run_calvin_unshielded_sequence(
                env, policy, task_oracle, lang_embeddings, initial_state, eval_sequence, val_annotations,
                get_env_state_for_initial_condition, ep_len=cfg.ep_len, replan_steps=REPLAN_STEPS,
                obstacles=None, record_trajectory=True,
            )
            n_done += 1
            if not attempts:
                log(f"[collect] {task_name} rollout {rollout_id}: no attempt recorded, not checkpointed")
                continue
            attempt = attempts[0]
            save_rollout_checkpoint(rollout_checkpoint_path(CHECKPOINT_DIR, task_name, rollout_id), {
                "rollout_id": rollout_id,
                "reached": attempt["reached"],
                "steps_taken": attempt["steps_taken"],
                "trajectory": attempt["trajectory"],
            })
            elapsed = time.perf_counter() - started
            eta_h = elapsed / n_done * (n_pending - n_done) / 3600
            log(f"[collect] {task_name} rollout {rollout_id}: reached={attempt['reached']} "
                f"steps_taken={attempt['steps_taken']} took={time.perf_counter() - t0:.1f}s "
                f"[{n_done}/{n_pending}, ETA {eta_h:.1f} h]")

        records = [
            load_rollout_checkpoint(rollout_checkpoint_path(CHECKPOINT_DIR, task_name, i))
            for i in range(N_ROLLOUTS)
            if os.path.exists(rollout_checkpoint_path(CHECKPOINT_DIR, task_name, i))
        ]
        if records:
            summary.append((task_name, _write_derived_files(task_name, records, cfg.ep_len, log), len(records)))

    log("[collect] summary (task, n_success/n_rollouts):")
    for task_name, n_success, n_records in summary:
        log(f"    {task_name:26s} {n_success}/{n_records}")
    log(f"[collect] checkpoints + raw/workspace files: {CHECKPOINT_DIR}")
    log.file.close()


if __name__ == "__main__":
    main()
