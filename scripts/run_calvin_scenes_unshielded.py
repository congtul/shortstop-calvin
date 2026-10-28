"""Unshielded MDT policy on the frozen obstacle scenes
(shortstop/calvin/data/scenes/<split>_scenes.json, built by
scripts/build_calvin_scene_manifest.py) -- the no-shield reference point
every shielded baseline is compared against on the same scenes.

Each scene runs once per OBSTACLE_MODES entry: reset to its stored initial
state, policy seeded with the scene's own `policy_seed`, one task (1 task
per sequence). "frozen" uses the scene's frozen obstacle (no live
placement); "none" runs the identical scene/seed with no obstacle -- the
success ceiling a shield could at best approach on these exact scenes. A
rollout ends at success, at the first violation, or at ep_len.

Before running a scene it also checks, in the real env, that the frozen
scene still holds: CALVIN reproduces the stored scene_obs
(scene_manifest.load_scenes), the blocks are where scene_obs says after the
reset, and the obstacle doesn't already touch the arm at the start
(experiment._clearance, the same ground truth the run uses). Any failure is
logged and the scene is counted but flagged.

No videos are recorded (cheap to run); to look at a scene afterwards, render
it from its scene_id in results.json:
    python scripts/render_obstacle_video.py +scene_id=<id> +baseline=unshielded

Run from WSL2, inside `mdt_env`:

    cd SHORTSTOP
    python scripts/run_calvin_scenes_unshielded.py                    # val + test
    python scripts/run_calvin_scenes_unshielded.py +splits=[pilot]    # any split(s)
"""
import sys
from pathlib import Path

import calvin_env  # noqa: F401 -- import side effects only, matches every other CALVIN script
import hydra
import numpy as np
from pytorch_lightning import seed_everything

REPO_ROOT = Path(__file__).resolve().parents[1]
MDT_POLICY_ROOT = REPO_ROOT / "mdt_policy"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(MDT_POLICY_ROOT))

from shortstop.calvin import scene_manifest as sm  # noqa: E402
from shortstop.calvin.baseline_runner import (  # noqa: E402
    make_logger, make_run_output_dir, setup_env_and_policy, write_results_json,
)
from shortstop.calvin.config import REPLAN_STEPS  # noqa: E402
from shortstop.calvin.experiment import _clearance, run_calvin_unshielded_sequence  # noqa: E402

SPLITS = ("val", "test")
OBSTACLE_MODES = ("frozen", "none")
SCENES_DIR = REPO_ROOT / "shortstop" / "calvin" / "data" / "scenes"
RUN_NAME = "calvin_scenes_unshielded_runs"


def _check_scene_in_env(env, scene):
    """Problems found after resetting to `scene` (empty list = fine)."""
    obs = env.reset(robot_obs=np.asarray(scene["robot_obs"]), scene_obs=np.asarray(scene["scene_obs"]))
    problems = []
    movable = env.get_info()["scene_info"]["movable_objects"]
    for name, i in sm._SCENE_OBS_BLOCK.items():
        drift = np.linalg.norm(np.asarray(movable[name]["current_pos"]) - np.asarray(scene["scene_obs"][i:i + 3]))
        if drift > 0.01:
            problems.append(f"{name} {drift:.3f} m from scene_obs after reset")
    start_clearance = _clearance(obs, scene["obstacles"])
    if start_clearance <= 0:
        problems.append(f"obstacle touches the arm at start (clearance {start_clearance:.3f})")
    return problems, start_clearance


def _rates(rows):
    n = len(rows)
    return {
        "n": n,
        "violation_rate": sum(r["violated"] for r in rows) / n if n else None,
        "success_rate": sum(r["reached"] for r in rows) / n if n else None,
        "timeout_rate": sum(not r["violated"] and not r["reached"] for r in rows) / n if n else None,
    }


@hydra.main(config_path="../mdt_policy/conf", config_name="mdt_evaluate")
def main(cfg):
    run_output_dir = make_run_output_dir(REPO_ROOT, RUN_NAME, tuning_mode=False)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    log = make_logger(run_output_dir / "run.log")
    log(f"[run] writing log + results.json to: {run_output_dir}")

    seed_everything(0, workers=True)
    env, policy, task_oracle, val_annotations, lang_embeddings = setup_env_and_policy(cfg, n_candidates=1)
    from mdt.evaluation.utils import get_env_state_for_initial_condition

    results = {"ep_len": cfg.ep_len, "replan_steps": REPLAN_STEPS, "splits": {}}
    splits = list(cfg.get("splits", SPLITS))
    for mode, split in [(m, s) for m in OBSTACLE_MODES for s in splits]:
        manifest, scenes = sm.load_scenes(SCENES_DIR / f"{split}_scenes.json", get_env_state_for_initial_condition)
        log(f"[run] split={split} obstacle={mode}: {len(scenes)} scenes (manifest created {manifest['created']})")
        rows = []
        for scene in scenes:
            problems, start_clearance = _check_scene_in_env(env, scene)
            for problem in problems:
                log(f"[run] WARNING {scene['scene_id']}: {problem}")
            obstacles = scene["obstacles"] if mode == "frozen" else None
            seed_everything(scene["policy_seed"], workers=True)
            attempts = run_calvin_unshielded_sequence(
                env, policy, task_oracle, lang_embeddings, scene["initial_state"], scene["eval_sequence"],
                val_annotations, get_env_state_for_initial_condition, ep_len=cfg.ep_len,
                replan_steps=REPLAN_STEPS, obstacles=obstacles,
            )
            attempt = attempts[0]
            rows.append({
                "scene_id": scene["scene_id"], "task_name": scene["task_name"], "policy_seed": scene["policy_seed"],
                "violated": bool(attempt["violated"]), "reached": bool(attempt["reached"]),
                "steps_taken": int(attempt["steps_taken"]),
                "min_clearance": None if attempt["min_clearance"] is None else float(attempt["min_clearance"]),
                "start_clearance": float(start_clearance), "scene_problems": problems,
                "workspace_rollout_hit_rate": scene["provenance"]["workspace_rollout_hit_rate"],
            })
            log(f"[run] {mode:6s} {scene['scene_id']} {scene['task_name']:24s} violated={attempt['violated']!s:5} "
                f"reached={attempt['reached']!s:5} steps={attempt['steps_taken']}")

        per_task = {}
        for row in rows:
            per_task.setdefault(row["task_name"], []).append(row)
        summary = _rates(rows)
        results["splits"][f"{split}_{mode}"] = {**summary, "per_task": {t: _rates(r) for t, r in sorted(per_task.items())},
                                    "scenes": rows}
        log(f"[run] {split} obstacle={mode}: violation_rate={summary['violation_rate']:.3f} success_rate={summary['success_rate']:.3f} "
            f"timeout_rate={summary['timeout_rate']:.3f} (n={summary['n']})")

    write_results_json(run_output_dir / "results.json", results)
    log(f"[run] DONE -- results in {run_output_dir / 'results.json'}")
    log.file.close()


if __name__ == "__main__":
    main()
