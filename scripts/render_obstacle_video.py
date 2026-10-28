"""Render one frozen scene (shortstop/calvin/data/scenes/) for one baseline to
an MP4 under outputs/render_obstacle_video/: same initial state, obstacle and
policy seed as the scene-based runs, so it shows exactly the rollout those
runs scored. Scene ids are in the manifests and in each run's results.

`+debug=true` also prints, per decision, each candidate's g(a) and predicted
TCP and checks that Select returned the best-scoring admissible candidate.

Run from WSL2, inside `mdt_env`:

    cd SHORTSTOP
    python scripts/render_obstacle_video.py +scene_id=val_0012 +baseline=stl_monitor
    python scripts/render_obstacle_video.py +scene_id=test_0003 +baseline=unshielded
    python scripts/render_obstacle_video.py +scene_id=val_0012 +baseline=shortstop +g_a=off
    python scripts/render_obstacle_video.py +scene_id=val_0032 +baseline=stl_monitor +fallback=freeze
    python scripts/render_obstacle_video.py +scene_id=probe_local_cluster_0010 +baseline=shortstop +repair_mode=gradient
"""
import sys
from pathlib import Path

import calvin_env  # noqa: F401 -- must be imported before mdt_policy/ goes on sys.path (it contains a calvin_env/ dir)
import hydra
import numpy as np
from pytorch_lightning import seed_everything

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "mdt_policy"))

import shortstop.calvin.experiment as calvin_experiment  # noqa: E402
from shortstop.calvin import scene_manifest  # noqa: E402
from shortstop.calvin.baseline_runner import (  # noqa: E402
    clearance_stats, fallback_rate, latency_stats, setup_env_and_policy, shield_activation_rate,
)
from shortstop.calvin.config import (  # noqa: E402
    CBF_ALPHA, CONF_THRESH_THRESHOLD, HORIZON_STEPS, MODEL_ERROR, N_CANDIDATES, REPLAN_STEPS, SHORTSTOP_PARAMS,
    STL_MONITOR_EPSILON, W_BAR,
)
from shortstop.calvin.obstacle_viz import save_sequence_video  # noqa: E402
from shortstop.calvin.reach import propagate_arm_tube  # noqa: E402
from shortstop.calvin.robot_geometry import point_to_box_distance  # noqa: E402
from shortstop.calvin.shield import (  # noqa: E402
    ArmCBFShield, ArmConfThreshShield, ArmMPCFilterShield, ArmRepairShield, ArmSTLMonitorShield,
)

OUT_DIR = REPO_ROOT / "outputs" / "render_obstacle_video"
SCENES_DIR = REPO_ROOT / "shortstop" / "calvin" / "data" / "scenes"

# name -> (n_candidates, shield factory or None)
BASELINES = {
    "unshielded": (1, None),
    "best_of_n": (N_CANDIDATES, lambda: ArmConfThreshShield(disagreement_threshold=float("inf"),
                                                            horizon_steps=HORIZON_STEPS)),
    "conf_thresh": (N_CANDIDATES, lambda: ArmConfThreshShield(disagreement_threshold=CONF_THRESH_THRESHOLD,
                                                              horizon_steps=HORIZON_STEPS)),
    "stl_monitor": (N_CANDIDATES, lambda: ArmSTLMonitorShield(obstacles=[], epsilon=STL_MONITOR_EPSILON,
                                                              horizon_steps=HORIZON_STEPS)),
    "mpc_filter": (N_CANDIDATES, lambda: ArmMPCFilterShield(obstacles=[], w_bar=W_BAR, model_error=MODEL_ERROR,
                                                            horizon_steps=HORIZON_STEPS)),
    "cbf": (N_CANDIDATES, lambda: ArmCBFShield(obstacles=[], w_bar=W_BAR, model_error=MODEL_ERROR, alpha=CBF_ALPHA,
                                               steps=REPLAN_STEPS)),
    "shortstop": (N_CANDIDATES, lambda mode=SHORTSTOP_PARAMS["repair_mode"]: ArmRepairShield(
        obstacles=[], w_bar=W_BAR, model_error=MODEL_ERROR, horizon_steps=HORIZON_STEPS,
        **{**SHORTSTOP_PARAMS, "repair_mode": mode})),
}


def _install_debug_prints(shield):
    """Print g(a) per candidate and check Select's choice; returns restore()."""
    original_scores = calvin_experiment.calvin_progress_scores
    original_select = shield.select

    def scores(joint_angles, candidates, goal_box):
        values = original_scores(joint_angles, candidates, goal_box)
        if goal_box is None:
            print("  [g(a)] no goal -> 0.0 for every candidate")
            return values
        print(f"  [g(a)] goal box min={np.round(goal_box[0], 3).tolist()} max={np.round(goal_box[1], 3).tolist()}")
        for i, chunk in enumerate(candidates):
            tip = propagate_arm_tube(joint_angles, chunk, w_bar=0.0, model_error=0.0)[-1]["fingertip"].b
            check = -point_to_box_distance(tip, *goal_box)
            print(f"    i={i} g(a)={values[i]:+.4f} (recomputed {check:+.4f}) tcp={np.round(tip, 3).tolist()}")
        return values

    def select(joint_angles, candidates, values):
        chunk, info = original_select(joint_angles, candidates, values)
        chosen = next((i for i, c in enumerate(candidates) if c is chunk), None)
        mask = info.get("admissible_mask")
        if info.get("fallback"):
            print(f"  [select] total fallback, mask={mask}")
        elif chosen is None:
            print("  [select] corrected chunk (not one of the candidates)")
        else:
            best = max((i for i, ok in enumerate(mask) if ok), key=lambda i: values[i])
            print(f"  [select] mask={mask} chose={chosen} best={best} -> {'OK' if chosen == best else 'MISMATCH'}")
        return chunk, info

    calvin_experiment.calvin_progress_scores = scores
    shield.select = select

    def restore():
        calvin_experiment.calvin_progress_scores = original_scores
        shield.select = original_select

    return restore


@hydra.main(config_path="../mdt_policy/conf", config_name="mdt_evaluate")
def main(cfg):
    scene_id = cfg.get("scene_id", None)
    baseline = cfg.get("baseline", "stl_monitor")
    use_progress = cfg.get("g_a", "on") == "on"
    fallback = cfg.get("fallback", "hold")
    debug = bool(cfg.get("debug", False))
    if scene_id is None:
        raise ValueError("pass +scene_id=<split>_<index>, e.g. +scene_id=val_0012")
    if baseline not in BASELINES:
        raise ValueError(f"baseline={baseline!r} not in {sorted(BASELINES)}")
    n_candidates, make_shield = BASELINES[baseline]

    seed_everything(0, workers=True)
    env, policy, task_oracle, val_annotations, lang_embeddings = setup_env_and_policy(cfg, n_candidates)
    from mdt.evaluation.utils import get_env_state_for_initial_condition

    split = str(scene_id).rsplit("_", 1)[0]
    _manifest, scenes = scene_manifest.load_scenes(SCENES_DIR / f"{split}_scenes.json",
                                                   get_env_state_for_initial_condition)
    scene = next((s for s in scenes if s["scene_id"] == scene_id), None)
    if scene is None:
        raise ValueError(f"scene_id={scene_id!r} not in {split}_scenes.json")
    print(f"[render] {scene_id} task={scene['task_name']} baseline={baseline} g(a)={'on' if use_progress else 'off'} "
          f"policy_seed={scene['policy_seed']} obstacles="
          f"{[(np.round(o.center, 3).tolist(), o.radius) for o in scene['obstacles']]}")

    seed_everything(scene["policy_seed"], workers=True)
    common = dict(ep_len=cfg.ep_len, replan_steps=REPLAN_STEPS, obstacles=scene["obstacles"], record_trajectory=True)
    if make_shield is None:
        attempts = calvin_experiment.run_calvin_unshielded_sequence(
            env, policy, task_oracle, lang_embeddings, scene["initial_state"], scene["eval_sequence"],
            val_annotations, get_env_state_for_initial_condition, **common,
        )
    else:
        repair_mode = cfg.get("repair_mode", None)
        shield = make_shield(repair_mode) if repair_mode else make_shield()
        restore = _install_debug_prints(shield) if debug else (lambda: None)
        try:
            attempts = calvin_experiment.run_calvin_shielded_sequence(
                env, policy, task_oracle, lang_embeddings, scene["initial_state"], scene["eval_sequence"],
                val_annotations, get_env_state_for_initial_condition, shield, horizon_steps=HORIZON_STEPS,
                use_progress=use_progress, fallback_controller=fallback, **common,
            )
        finally:
            restore()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = [{
        "subtask": subtask, "state_snapshots": a["state_snapshots"], "obstacles": a["obstacles"],
        "outcome": "violated" if a["violated"] else "reached" if a["reached"] else "failed",
    } for subtask, a in zip(scene["eval_sequence"], attempts)]
    suffix = "" if make_shield is None else f"_{fallback}"
    if cfg.get("repair_mode", None):
        suffix += f"_{cfg.repair_mode}"
    out_path = OUT_DIR / f"{scene_id}_{baseline}{suffix}{'' if use_progress else '_no_g_a'}.mp4"
    save_sequence_video(env, records, str(out_path), save_violation_frame_dir=OUT_DIR)

    for subtask, a in zip(scene["eval_sequence"], attempts):
        print(f"  {subtask}: violated={a['violated']} reached={a['reached']} steps={a['steps_taken']} "
              f"min_clearance={a['min_clearance']}")
    if make_shield is not None:
        results = [attempts]
        latency = latency_stats(results)
        print(f"  activation={shield_activation_rate(results)} fallback={fallback_rate(results)} "
              f"select_ms_mean={None if latency is None else round(latency['mean'], 2)}")
    clearance = clearance_stats([attempts])
    if clearance is not None:
        print(f"  min_clearance={clearance['min']:.4f}")
    print(f"[render] wrote {out_path}")


if __name__ == "__main__":
    main()
