"""Is CALVIN's env.step deterministic for a given (state, action)?

Snapshots the env at reachable states of real rollouts, executes the same
already-sampled chunk N_REPEATS times from that snapshot (the policy is
sampled once, so only the env can differ) and prints the spread of the arm
frames between repeats. All zero (it is) means there is no disturbance
term: w_bar = 0, and the prediction error is all model_error.

Run from WSL2, inside `mdt_env`:

    cd SHORTSTOP
    python scripts/verify_calvin_env_determinism.py
"""
import sys
from datetime import datetime
from pathlib import Path

import calvin_env
import hydra
import numpy as np
from pytorch_lightning import seed_everything

REPO_ROOT = Path(__file__).resolve().parents[1]
MDT_POLICY_ROOT = REPO_ROOT / "mdt_policy"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(MDT_POLICY_ROOT))

from mdt.evaluation.multistep_sequences import get_sequences  # noqa: E402
from mdt.evaluation.utils import get_default_beso_and_env, get_env_state_for_initial_condition  # noqa: E402
from mdt.utils.utils import get_last_checkpoint  # noqa: E402

from shortstop.calvin.config import N_CANDIDATES, REPLAN_STEPS  # noqa: E402
from shortstop.calvin.experiment import (  # noqa: E402
    _joint_angles_from_obs, _lang_goal, _to_action_tensor, restore_env_snapshot, snapshot_env,
)
from shortstop.calvin.mdt_policy_client import ForwardOnlyPolicy  # noqa: E402
from shortstop.calvin.robot_geometry import panda_frames  # noqa: E402

# Same recipe as calibrate_arm_model_error.py -- applied here only if real
# residuals turn out nonzero (see module docstring).
QUANTILE = 0.99
SAFETY_FACTOR = 1.25

# Small spread of real (state, chunk) pairs is enough for this check --
# same "10 sequences, one subtask each" budget verify_robot_geometry_
# against_pybullet.py already uses for its own geometry cross-check.
N_SEQUENCES_TO_SAMPLE = 10
CHECKS_PER_SUBTASK = 3  # different (state, chunk) points tested along one subtask's own rollout
N_REPEATS = 3  # how many times the SAME chunk is re-run from the SAME snapshot per check

RUN_OUTPUT_DIR = REPO_ROOT / "outputs" / "verify_calvin_env_determinism_runs" / f"run_{datetime.now():%Y%m%d_%H%M%S}"

_LOG_FILE = None


def _log(msg):
    print(msg)
    if _LOG_FILE is not None:
        _LOG_FILE.write(msg + "\n")
        _LOG_FILE.flush()


def _residual_percentiles(residuals):
    values = np.asarray(residuals)
    return {
        "n": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)), "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)), "min": float(values.min()), "max": float(values.max()),
    }


def _run_chunk_and_record_joint_trajectory(env, chunk):
    """Executes every row of `chunk` via real env.step() calls, returns the
    resulting real joint angles after each row, stacked (len(chunk),
    N_JOINTS)."""
    trajectory = []
    for action_row in chunk:
        obs, _, _, _ = env.step(_to_action_tensor(action_row))
        trajectory.append(_joint_angles_from_obs(obs))
    return np.stack(trajectory)


def _repeat_chunk_from_same_snapshot(env, chunk, n_repeats):
    """Snapshots the CURRENT real env state once, then executes the exact
    SAME `chunk` (never re-sampled) `n_repeats` times from that identical
    snapshot, restoring before each repeat -- see module docstring for why
    this isolates env determinism from the policy's own per-call sampling
    noise. Leaves `env` restored to the pre-check snapshot when done (the
    caller is responsible for actually advancing the rollout afterward, by
    executing `chunk` once for real -- see main()'s own loop). Returns a
    list of `n_repeats` (len(chunk), N_JOINTS) joint-trajectory arrays."""
    snapshot = snapshot_env(env)
    trajectories = []
    for _ in range(n_repeats):
        restore_env_snapshot(env, snapshot)
        trajectories.append(_run_chunk_and_record_joint_trajectory(env, chunk))
    restore_env_snapshot(env, snapshot)
    return trajectories


def _residuals_from_repeats(trajectories):
    """Max-per-frame Cartesian distance (meters) -- same convention as
    shortstop.calvin.reach.step_prediction_residual -- between the FIRST
    repeat's real joint trajectory and every OTHER repeat's, at every
    step. Would be exactly 0 at every entry if the environment is
    deterministic given (state, action) -- there is no nominal MODEL
    involved in this comparison at all, both sides are real env.step()
    outcomes, so any nonzero value here is genuine environment
    stochasticity (w_t), never model approximation error."""
    reference = trajectories[0]
    residuals = []
    for other in trajectories[1:]:
        for step in range(len(reference)):
            frames_ref = panda_frames(reference[step])
            frames_other = panda_frames(other[step])
            residuals.append(float(np.max(np.linalg.norm(frames_ref - frames_other, axis=1))))
    return residuals


@hydra.main(config_path="../mdt_policy/conf", config_name="mdt_evaluate")
def main(cfg):
    global _LOG_FILE
    RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _LOG_FILE = open(RUN_OUTPUT_DIR / "run.log", "w", encoding="utf-8")
    _log(f"[run] writing log to: {RUN_OUTPUT_DIR}")
    _log(
        f"[run] config: n_sequences={N_SEQUENCES_TO_SAMPLE} checks_per_subtask={CHECKS_PER_SUBTASK} "
        f"n_repeats={N_REPEATS} replan_steps={REPLAN_STEPS}"
    )

    seed_everything(0, workers=True)

    checkpoint = get_last_checkpoint(Path(cfg.train_folder))
    model, env, _, lang_embeddings = get_default_beso_and_env(
        cfg.train_folder, cfg.dataset_path, checkpoint,
        eval_cfg_overwrite=cfg.eval_cfg_overwrite, device_id=cfg.device,
    )
    model.num_sampling_steps = cfg.num_sampling_steps
    model.sampler_type = cfg.sampler_type
    model.eval()

    task_oracle = hydra.utils.instantiate(cfg.tasks)
    val_annotations = cfg.annotations
    # Same n_candidates the real deployed pipeline uses -- only candidates[0]
    # is ever touched below, but matching the real batch size keeps the
    # diffusion sampler in the same regime a real run would exercise.
    policy = ForwardOnlyPolicy(model, n_candidates=N_CANDIDATES)

    eval_sequences = get_sequences(2 * N_SEQUENCES_TO_SAMPLE)[0:N_SEQUENCES_TO_SAMPLE]

    all_residuals = []
    for idx, (initial_state, eval_sequence) in enumerate(eval_sequences):
        seed_everything(1000 + idx, workers=True)
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

        for subtask in eval_sequence[:1]:  # one subtask per sequence is enough spread, see module docstring
            obs = env.get_obs()
            goal = _lang_goal(lang_embeddings, val_annotations, subtask)
            for check_i in range(CHECKS_PER_SUBTASK):
                candidates = policy.propose({**obs, "goal": goal})
                chunk = candidates[0][:REPLAN_STEPS]

                trajectories = _repeat_chunk_from_same_snapshot(env, chunk, N_REPEATS)
                residuals = _residuals_from_repeats(trajectories)
                all_residuals.extend(residuals)
                _log(
                    f"  seq{idx} subtask={subtask} check={check_i}: "
                    f"max_residual={max(residuals):.8f}  mean_residual={float(np.mean(residuals)):.8f}"
                )

                # Advance the REAL rollout by executing `chunk` for real
                # exactly once (env is currently sitting at the pre-check
                # snapshot, per _repeat_chunk_from_same_snapshot's own
                # contract), so the next check_i samples a genuinely new
                # state instead of re-testing the same one.
                for action_row in chunk:
                    obs, _, _, _ = env.step(_to_action_tensor(action_row))
            break  # one subtask per sequence is plenty, see module docstring

    stats = _residual_percentiles(all_residuals)
    _log(
        f"[residuals] n={stats['n']}  mean={stats['mean']:.8f}  median={stats['median']:.8f}  "
        f"p10={stats['p10']:.8f}  p90={stats['p90']:.8f}  p99={stats['p99']:.8f}  "
        f"min={stats['min']:.8f}  max={stats['max']:.8f}"
    )

    if stats["max"] < 1e-9:
        _log(
            "[result] every repeat was bit-for-bit identical (max residual below floating-point "
            "noise) -- CALVIN's env.step() is DETERMINISTIC given an identical (state, action) "
            "pair for every sample tested here. w_bar=0.0 (already the default for every real "
            "ShortStop-family CALVIN shield) is CONFIRMED, not merely assumed -- there is no "
            "measurable w_t in this simulated system; model_error alone correctly carries the "
            "full budget."
        )
    else:
        calibrated_w_bar = stats["p99"] * SAFETY_FACTOR
        _log(
            f"[result] repeats were NOT identical -- real environment stochasticity detected. "
            f"Calibrated w_bar = p99 * {SAFETY_FACTOR} = {calibrated_w_bar:.8f} (same Table VII "
            f"recipe as model_error). This is a genuine, SEPARATE quantity from model_error (no "
            f"nominal model was involved in this measurement at all) -- consider using it "
            f"ALONGSIDE the calibrated model_error, not instead of it, in whichever Arm*Shield "
            f"construction currently hardcodes w_bar=0.0."
        )
    _log(f"[run] DONE -- see {RUN_OUTPUT_DIR / 'run.log'} for the full per-check breakdown.")
    _LOG_FILE.close()


if __name__ == "__main__":
    main()
