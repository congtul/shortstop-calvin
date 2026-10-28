"""Build the frozen obstacle scenes (one task per scene) into
shortstop/calvin/data/scenes/<split>_scenes.json; see
shortstop.calvin.scene_manifest for the rules. CPU only, needs the workspace
clouds (collect_calvin_unshielded_rollouts.py) and the static-target cache
(precompute_static_target_sweeps.py).

Initial states are split once into disjoint val/test pools (PARTITION_SEED,
VAL_FRACTION). Each scene has its own RNG stream and policy seed, so scene i
does not change with the number of scenes. Optional args: hit-rate band
(lo hi) and center offset (m).

Run from WSL2, inside `mdt_env`:

    cd SHORTSTOP
    python scripts/build_calvin_scene_manifest.py val 50 0.05 0.4 0.2
    python scripts/build_calvin_scene_manifest.py test 50 0.05 0.4 0.2
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MDT_POLICY_ROOT = REPO_ROOT / "mdt_policy"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(MDT_POLICY_ROOT))

from shortstop.calvin import scene_manifest as sm  # noqa: E402
from shortstop.calvin import feasibility_planner as fp  # noqa: E402
from shortstop.calvin.config import OBSTACLE_RADIUS  # noqa: E402
from shortstop.calvin.independent_tasks import _enumerate_initial_states  # noqa: E402
from shortstop.calvin.robot_geometry import to_world_frame  # noqa: E402
from shortstop.calvin.rollout_collection import eligible_task_names, records_from_raw, workspace_arrays  # noqa: E402

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "val"   # "val" | "test" | "pilot" | "final" | "probe_<family>"
# pilot and probes draw from val's pool; final from every initial state no val/test/probe_local_cluster scene used.
POOL = {"test": "test", "final": "unused"}.get(SPLIT, "val")
# probe_<family>: obstacle layouts from scene_manifest.FAMILIES (several obstacles, bars, near-target).
# final: the local_cluster rules of its tuning val (probe_local_cluster).
FAMILY = SPLIT[len("probe_"):] if SPLIT.startswith("probe_") else "local_cluster" if SPLIT == "final" else None
TOTAL_SCENES = int(sys.argv[2]) if len(sys.argv) > 2 else 50
# Optional difficulty band on the obstacle's reference-rollout hit rate
# (scene_manifest.sample_scene_obstacle's hit_rate_range); None = no band.
HIT_RATE_RANGE = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else None
# Optional max random offset (m) of the obstacle center from its workspace
# point (sample_scene_obstacle's center_offset_max); 0 = on the point.
CENTER_OFFSET_MAX = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0

PARTITION_SEED = 0
VAL_FRACTION = 0.5
FAMILY_INDEX = list(sm.FAMILIES).index(FAMILY) if FAMILY else None
SCENE_SEED = {"pilot": 0, "val": 1, "test": 2, "final": 3}.get(SPLIT, 100 + (FAMILY_INDEX or 0))
# The final split's workspace cloud and hit rate include the hand (flange ->
# TCP); the pivot splits were built without it and stay reproducible.
INCLUDE_HAND = SPLIT == "final" or FAMILY is not None
# Per-scene eval seed = base + scene index; disjoint from eval scripts
# (1000+), rollout collection (2_000_000+) and every other split.
POLICY_SEED_BASE = {"pilot": 4_000_000, "val": 5_000_000, "test": 6_000_000, "final": 7_000_000}.get(
    SPLIT, 8_000_000 + 100_000 * (FAMILY_INDEX or 0))

# Retries with a fresh initial state when no obstacle candidate passes.
MAX_STATE_RETRIES = 5

WORKSPACE_DIR = REPO_ROOT / "outputs" / "unshielded_rollouts"
TASKS_YAML = MDT_POLICY_ROOT / "calvin_env" / "conf" / "tasks" / "new_playtable_tasks.yaml"
OUT_PATH = REPO_ROOT / "shortstop" / "calvin" / "data" / "scenes" / f"{SPLIT}_scenes.json"


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    from mdt.evaluation.multistep_sequences import tasks, valid_task
    from mdt.evaluation.utils import get_env_state_for_initial_condition

    task_args = {name: tuple(spec) for name, spec in yaml.safe_load(TASKS_YAML.read_text())["tasks"].items()}
    initial_states = _enumerate_initial_states()
    eligible, _ = eligible_task_names(tasks, initial_states, valid_task)
    task_names = [name for name in eligible if name not in sm.NO_OBSTACLE_TASKS]
    val_pool, test_pool = sm.partition_states(initial_states, VAL_FRACTION, PARTITION_SEED)
    if POOL == "unused":
        used = set()
        for split in ("val", "test", "probe_local_cluster"):  # probe_local_cluster = the final's tuning val
            pivot = json.loads((OUT_PATH.parent / f"{split}_scenes.json").read_text())
            used |= {scene["initial_state_index"] for scene in pivot["scenes"]}
        pool = [i for i in range(len(initial_states)) if i not in used]
    else:
        pool = val_pool if POOL == "val" else test_pool
    allocation = sm.allocate_trials(task_names, TOTAL_SCENES, SCENE_SEED)

    workspaces = {}
    for name in task_names:
        data = np.load(WORKSPACE_DIR / f"workspace_{name}.npz")
        raw = np.load(WORKSPACE_DIR / f"rollouts_{name}.npz")
        points = data["points"]
        if INCLUDE_HAND:
            points = workspace_arrays(records_from_raw(raw), int(data["link_samples_per_segment"]),
                                      bool(data["only_successful"]), include_hand=True)["points"]
        workspaces[name] = {"points": points, "n_rollouts": int(data["n_rollouts"]),
                            "n_success": int(data["n_success"]), "frames": raw["frames"],
                            "rollout_id": raw["rollout_id"]}

    # move_door_rel tasks: the measured contact point on the moving link
    # (scene_manifest.door_contact_offset), from each task's own rollouts.
    door_offsets, door_offset_info = {}, {}
    for name in task_names:
        kind, *args = task_args[name]
        if kind != "move_door_rel":
            continue
        joint, threshold = args
        starts = {float(get_env_state_for_initial_condition(initial_states[s])[1][sm._SCENE_OBS_JOINT[joint]])
                  for s in range(len(initial_states)) if valid_task(initial_states[s], tasks[name])}
        if len(starts) != 1:
            raise ValueError(f"{name}: expected one start value for {joint}, got {sorted(starts)}")
        q_end = starts.pop() + threshold
        raw = np.load(WORKSPACE_DIR / f"rollouts_{name}.npz")
        offset, std, n = sm.door_contact_offset(raw, joint, q_end)
        door_offsets[name] = offset
        door_offset_info[name] = {"offset_from_box_min": offset, "std": std, "n_successful": n, "q_end": q_end}
        print(f"[build] {name}: contact offset {np.round(offset, 4).tolist()} (std {np.round(std, 4).tolist()}, n={n})")

    print(f"[build] split={SPLIT} pool={POOL} ({len(pool)} states) total={TOTAL_SCENES} "
          f"tasks={len(task_names)} obstacle_radius={OBSTACLE_RADIUS}")
    scenes, failures = [], []
    started = time.perf_counter()
    for i, task_name in enumerate(allocation):
        rng = np.random.default_rng([SCENE_SEED, i])
        planner_rng = np.random.default_rng([SCENE_SEED, i, 1])
        valid = [s for s in pool if valid_task(initial_states[s], tasks[task_name])]
        obstacle = None
        for attempt in range(MAX_STATE_RETRIES):
            state_index = int(valid[rng.integers(len(valid))])
            initial_state = initial_states[state_index]
            robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
            goals = sm.task_goals(task_name, task_args[task_name], scene_obs, door_offsets)
            if FAMILY is not None:
                obstacles, info = sm.sample_scene_obstacles(
                    sm.FAMILIES[FAMILY], workspaces[task_name]["points"], goals, rng, planner_rng,
                    workspaces[task_name]["frames"], workspaces[task_name]["rollout_id"], include_hand=True,
                )
            else:
                obstacle, info = sm.sample_scene_obstacle(
                    workspaces[task_name]["points"], goals, OBSTACLE_RADIUS, rng, planner_rng,
                    max_candidates=sm.MAX_OBSTACLE_CANDIDATES if HIT_RATE_RANGE is None else 300,
                    hit_rate_range=HIT_RATE_RANGE, center_offset_max=CENTER_OFFSET_MAX,
                    rollout_frames=workspaces[task_name]["frames"],
                    rollout_id=workspaces[task_name]["rollout_id"], include_hand=INCLUDE_HAND,
                )
                obstacles = None if obstacle is None else [obstacle]
            if obstacles is not None:
                break
            print(f"[build] scene {i} {task_name}: state {state_index} rejected all candidates {info['rejected']}")
        if obstacles is None:
            failures.append({"scene_index": i, "task_name": task_name})
            continue

        contact, completion = goals
        planner = info.pop("planner")
        scenes.append(_jsonable({
            "scene_id": f"{SPLIT}_{i:04d}",
            "split": SPLIT,
            "task_name": task_name,
            "initial_state_index": state_index,
            "initial_state": initial_state,
            "robot_obs": robot_obs,
            "scene_obs": scene_obs,
            "obstacle_center_local": obstacles[0].center,
            "obstacle_center_world": to_world_frame(obstacles[0].center, sm.BASE_POSITION, sm.BASE_ORIENTATION),
            "obstacle_radius": obstacles[0].radius,
            "obstacles": [{"center_local": o.center, "radius": o.radius,
                           "center_world": to_world_frame(o.center, sm.BASE_POSITION, sm.BASE_ORIENTATION)}
                          for o in obstacles],
            "policy_seed": POLICY_SEED_BASE + i,
            "provenance": {
                **info,
                "workspace_rollout_hit_rate": sm.rollout_hit_rate(
                    workspaces[task_name]["frames"], workspaces[task_name]["rollout_id"], obstacles, INCLUDE_HAND),
                "state_retries": attempt,
                "contact_goal": {"box_min": contact.box_min, "box_max": contact.box_max, **contact.meta},
                "completion_goal": ({"wrist_turn": completion.wrist_turn, **completion.meta}
                                    if completion.wrist_turn is not None else
                                    {"box_min": completion.box_min, "box_max": completion.box_max, **completion.meta}),
                "planner_attempts": planner["attempts"],
                "planner_contact_config": planner["contact_config"],
                "planner_completion_config": planner["completion_config"],
            },
        }))
        print(f"[build] scene {i:3d} {task_name:24s} state={state_index:3d} candidates={info['candidates_drawn']:2d} "
              f"hit_rate={scenes[-1]['provenance']['workspace_rollout_hit_rate']:.2f} "
              f"[{time.perf_counter() - started:.0f}s]")

    manifest = _jsonable({
        "version": sm.MANIFEST_VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "split": SPLIT,
        "pool": POOL,
        "settings": {
            "total_scenes": TOTAL_SCENES, "partition_seed": PARTITION_SEED, "val_fraction": VAL_FRACTION,
            "scene_seed": SCENE_SEED, "policy_seed_base": POLICY_SEED_BASE, "obstacle_radius": OBSTACLE_RADIUS,
            "max_obstacle_candidates": sm.MAX_OBSTACLE_CANDIDATES, "max_state_retries": MAX_STATE_RETRIES,
            "planner_margin": fp.PLANNER_MARGIN, "min_start_clearance": sm.MIN_START_CLEARANCE,
            "hit_rate_range": HIT_RATE_RANGE, "workspace_includes_hand": INCLUDE_HAND,
            "family": None if FAMILY is None else {"name": FAMILY, **sm.FAMILY_DEFAULTS, **sm.FAMILIES[FAMILY]}, "center_offset_max": CENTER_OFFSET_MAX, "contact_tol": fp.CONTACT_TOL, "down_cone_deg": fp.DOWN_CONE_DEG,
            "lift_extra": sm.LIFT_EXTRA, "base_position": sm.BASE_POSITION, "frame": "robot_base_local",
        },
        "partition": {"val_state_indices": val_pool, "test_state_indices": test_pool, "pool_state_indices": pool},
        "excluded_tasks": list(sm.NO_OBSTACLE_TASKS),
        "door_contact_offsets": door_offset_info,
        "workspace": {name: {k: w[k] for k in ("n_rollouts", "n_success")} for name, w in workspaces.items()},
        "failures": failures,
        "scenes": scenes,
    })
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sm.save_manifest(OUT_PATH, manifest)
    per_task = {}
    for scene in scenes:
        per_task[scene["task_name"]] = per_task.get(scene["task_name"], 0) + 1
    print(f"[build] wrote {len(scenes)} scenes ({len(failures)} failed) to {OUT_PATH} "
          f"in {time.perf_counter() - started:.0f}s")
    print(f"[build] per task: {dict(sorted(per_task.items()))}")


if __name__ == "__main__":
    main()
