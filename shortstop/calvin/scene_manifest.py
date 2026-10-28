"""Frozen CALVIN obstacle scenes, built once by
scripts/build_calvin_scene_manifest.py and read by every run.

A scene = (task, initial state, obstacle, policy seed). The obstacle:
- is drawn from the task's workspace cloud (arm geometry of separately
  seeded unshielded rollouts, scripts/collect_calvin_unshielded_rollouts.py),
  optionally shifted by up to `center_offset_max`, so it sits where the
  policy tends to go;
- keeps MIN_START_CLEARANCE from the arm at the start pose and stays off the
  task's contact and completion goals;
- has a rollout hit rate inside the difficulty band, when one is given;
- leaves a collision-free path start -> contact -> completion
  (shortstop.calvin.feasibility_planner), so every scene is solvable.

Val and test draw from disjoint pools of initial states. Everything here is
offline: CALVIN derives scene_obs deterministically from the initial state,
so goals are known without the simulator (load_scenes re-checks scene_obs).
Obstacles and goals are in the robot-base local frame.
"""
import json

import numpy as np

from .config import SCENE_TARGET_MARGIN
from .feasibility_planner import PLANNER_MARGIN, TaskGoal, arm_obstacle_clearance, check_feasibility
from .robot_geometry import (
    GRIPPER_TIP_OFFSET, GRIPPER_TIP_RADIUS, LINK_RADIUS, gripper_tip_position, point_to_box_distance, to_world_frame,
)
from .static_targets import static_target_box_at
from .rollout_collection import tcp_from_frames
from .task_goals import LIFT_EXTRA, task_goals as shared_task_goals  # noqa: F401 -- LIFT_EXTRA is recorded by the builder
from ..env import Obstacle

MANIFEST_VERSION = 1

# calvin_scene_D's robot base pose (FK of START_JOINT_ANGLES with it matches
# CALVIN's reset TCP to < 1 um).
BASE_POSITION = (-0.34, -0.46, 0.24)
BASE_ORIENTATION = (0.0, 0.0, 0.0, 1.0)

# Joint angles CALVIN resets to (the same for every initial state).
START_JOINT_ANGLES = np.array([-1.21779124, 1.03987629, 2.11978254, -2.34205014, -0.87015899, 1.64119093, 0.55344928])

# scene_obs layout (get_env_state_for_initial_condition).
_SCENE_OBS_JOINT = {"base__slide": 0, "base__drawer": 1, "base__button": 2, "base__switch": 3}
_SCENE_OBS_BLOCK = {"block_red": 6, "block_blue": 12, "block_pink": 18}


# Minimum obstacle-to-arm clearance (m) at the start pose. With only the
# planner's 1 cm, 12/100 unshielded runs hit within the first 4 steps,
# before a shield's second decision.
MIN_START_CLEARANCE = 0.05

# Workspace draws per scene before giving up on it.
MAX_OBSTACLE_CANDIDATES = 60

# Tasks with no valid obstacle placement under these rules (none at present).
NO_OBSTACLE_TASKS = ()


def scene_info_from_scene_obs(scene_obs):
    """The parts of env.get_info()['scene_info'] that task_goals reads,
    rebuilt from a scene_obs."""
    scene_obs = np.asarray(scene_obs, dtype=float)
    return {
        "movable_objects": {name: {"current_pos": scene_obs[i:i + 3].copy()} for name, i in _SCENE_OBS_BLOCK.items()},
        "doors": {joint: {"current_state": float(scene_obs[_SCENE_OBS_JOINT[joint]])}
                  for joint in ("base__slide", "base__drawer")},
        "buttons": {"base__button": {"joint_state": float(scene_obs[_SCENE_OBS_JOINT["base__button"]])}},
        "switches": {"base__switch": {"joint_state": float(scene_obs[_SCENE_OBS_JOINT["base__switch"]])}},
        "fixed_objects": {"table": {}},
    }


def fingertip_from_frames(frames):
    """TCP from recorded panda_frames() rows (..., 9, 3)."""
    return tcp_from_frames(frames)

def door_contact_offset(rollouts, joint, q_end):
    """Where a move_door_rel task's fingertip works the door: the mean, over
    the task's successful rollouts (`rollouts` = rollouts_<task>.npz), of
    the last-step fingertip minus the link box_min at `q_end`, in world
    coordinates. Returns (offset, per-axis std, n_successful)."""
    frames, rollout_id, reached = rollouts["frames"], rollouts["rollout_id"], rollouts["per_rollout_reached"]
    ids = np.asarray(rollouts["per_rollout_id"])[np.asarray(reached, dtype=bool)]
    if len(ids) == 0:
        raise ValueError("no successful rollouts to measure the contact point from")
    last = np.array([frames[rollout_id == k][-1] for k in ids])
    local = fingertip_from_frames(last)
    tips = np.array([to_world_frame(t, BASE_POSITION, BASE_ORIENTATION) for t in local])
    box_min, _ = _required_box(joint, q_end)
    offsets = tips - box_min
    return offsets.mean(axis=0), offsets.std(axis=0), len(ids)


def task_goals(task_name, task_args, scene_obs, door_contact_offsets=None):
    """Shared task_goals.task_goals for a scene's state with the robot at its
    reset pose; `task_args` = (function name, *config args)."""
    return shared_task_goals(
        task_name, task_args, scene_info_from_scene_obs(scene_obs), gripper_tip_position(START_JOINT_ANGLES),
        BASE_POSITION, BASE_ORIENTATION, door_contact_offsets,
    )


def _required_box(name, q):
    box = static_target_box_at(name, q)
    if box is None:
        raise ValueError(f"no cached endpoint boxes for {name!r} -- run scripts/precompute_static_target_sweeps.py")
    return box


def obstacle_clears_targets(center, radius, goals):
    """Does the obstacle keep radius + GRIPPER_TIP_RADIUS + SCENE_TARGET_MARGIN
    from every positional goal?"""
    need = radius + GRIPPER_TIP_RADIUS + SCENE_TARGET_MARGIN
    return all(
        point_to_box_distance(center, g.box_min, g.box_max) >= need
        for g in goals if g.box_min is not None
    )


def partition_states(initial_states, val_fraction, rng_seed):
    """Disjoint (val_indices, test_indices) into `initial_states`."""
    order = np.random.RandomState(rng_seed).permutation(len(initial_states))
    n_val = int(round(val_fraction * len(initial_states)))
    return sorted(order[:n_val].tolist()), sorted(order[n_val:].tolist())


def allocate_trials(task_names, total, rng_seed):
    """`total` trials spread as evenly as possible over `task_names`; the
    tasks that get one extra are chosen by a seeded shuffle. Returns a list
    of task names (length `total`) in round-robin order."""
    order = list(np.random.RandomState(rng_seed).permutation(len(task_names)))
    return [task_names[order[i % len(task_names)]] for i in range(total)]


def sample_scene_obstacle(workspace_points, goals, radius, rng, planner_rng, max_candidates=MAX_OBSTACLE_CANDIDATES,
                          hit_rate_range=None, rollout_frames=None, rollout_id=None, center_offset_max=0.0,
                          include_hand=False):
    """First workspace-drawn obstacle passing every scene rule (see module
    docstring); returns (Obstacle or None, info with rejection counts and
    planner stats).

    `hit_rate_range` (lo, hi): keep obstacles whose `rollout_hit_rate` over
    the reference rollouts is in [lo, hi) -- the difficulty knob (hit < 0.5
    gave 29% unshielded violations, >= 0.8 gave 98%). The reference rollouts
    use their own seeds, never the eval seeds.
    `center_offset_max` (m): move the drawn point by Uniform(0, max) in a
    random direction; points on a trajectory are hit by most rollouts, so
    the offset is what fills the low end of the band."""
    contact, completion = goals
    rejected = {"start_clearance": 0, "target_keepout": 0, "hit_rate": 0, "infeasible": 0}
    start_clearance = None
    for candidate in range(1, max_candidates + 1):
        point_index = int(rng.integers(len(workspace_points)))
        center = np.asarray(workspace_points[point_index], dtype=float)
        if center_offset_max > 0.0:
            direction = rng.normal(size=3)
            center = center + rng.uniform(0.0, center_offset_max) * direction / np.linalg.norm(direction)
        obstacle = Obstacle(center=center, radius=radius)
        start_clearance = arm_obstacle_clearance(START_JOINT_ANGLES, [obstacle])
        if start_clearance < max(MIN_START_CLEARANCE, PLANNER_MARGIN):
            rejected["start_clearance"] += 1
            continue
        if not obstacle_clears_targets(center, radius, goals):
            rejected["target_keepout"] += 1
            continue
        if hit_rate_range is not None:
            hit_rate = rollout_hit_rate(rollout_frames, rollout_id, obstacle, include_hand)
            if not hit_rate_range[0] <= hit_rate < hit_rate_range[1]:
                rejected["hit_rate"] += 1
                continue
        result = check_feasibility(START_JOINT_ANGLES, contact, completion, [obstacle], planner_rng)
        if not result["feasible"]:
            rejected["infeasible"] += 1
            continue
        return obstacle, {"candidates_drawn": candidate, "workspace_point_index": point_index,
                          "start_clearance": float(start_clearance),
                          "rejected": rejected, "planner": result}
    return None, {"candidates_drawn": max_candidates, "rejected": rejected}


# Obstacle-layout families for scene building (see sample_scene_obstacles):
# n = number of placed obstacles (a bar counts as one), radius range (m),
# bar_prob = chance an obstacle is a bar of `bar_spheres` spheres in a line,
# near_target_prob = chance it is placed `near_target` (m) from the contact
# goal instead of drawn from the workspace, hit_rate_range on the union.
FAMILIES = {
    "single5": dict(n=(1, 1), radius=(0.05, 0.05), bar_prob=0.0, near_target_prob=0.0),
    "clutter3": dict(n=(3, 3), radius=(0.03, 0.07), bar_prob=0.0, near_target_prob=0.0),
    "near_target": dict(n=(2, 2), radius=(0.04, 0.06), bar_prob=0.0, near_target_prob=0.5),
    "bar": dict(n=(1, 1), radius=(0.03, 0.03), bar_prob=1.0, near_target_prob=0.0),
    "mixed": dict(n=(2, 3), radius=(0.03, 0.06), bar_prob=0.3, near_target_prob=0.3),
    # Local clutter: 1-3 separate spheres within cluster_radius of one anchor,
    # the anchor near the contact goal or on the workspace path; no overlap;
    # the feasibility path must keep planner_margin (>= MODEL_ERROR) so a
    # certified shield can follow it.
    "local_cluster": dict(n=(1, 3), radius=(0.03, 0.07), bar_prob=0.0, near_target_prob=0.5,
                          cluster_radius=0.20, min_gap=0.02, planner_margin=0.02),
}
FAMILY_DEFAULTS = dict(bar_spheres=5, near_target=(0.13, 0.25), center_offset_max=0.20, hit_rate_range=(0.2, 0.8))


def _random_unit(rng):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def _draw_local_cluster(family, workspace_points, goals, rng):
    """1-3 non-overlapping spheres within cluster_radius of one anchor (near
    the contact goal or on the workspace path); None if they overlap."""
    contact = goals[0]
    if rng.random() < family["near_target_prob"] and contact.box_min is not None:
        anchor = 0.5 * (np.asarray(contact.box_min) + np.asarray(contact.box_max))
        anchor = anchor + rng.uniform(*family["near_target"]) * _random_unit(rng)
    else:
        anchor = np.asarray(workspace_points[int(rng.integers(len(workspace_points)))], dtype=float)
        anchor = anchor + rng.uniform(0.0, family["center_offset_max"]) * _random_unit(rng)
    n = int(rng.integers(family["n"][0], family["n"][1] + 1))
    spheres = [Obstacle(anchor + rng.uniform(0.0, family["cluster_radius"]) * _random_unit(rng),
                        float(rng.uniform(*family["radius"]))) for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            gap = np.linalg.norm(spheres[i].center - spheres[j].center) - spheres[i].radius - spheres[j].radius
            if gap < family["min_gap"]:
                return None
    return spheres


def _draw_obstacle_group(family, workspace_points, goals, rng):
    """One obstacle (a sphere, or a bar as several spheres) per the family."""
    contact = goals[0]
    radius = float(rng.uniform(*family["radius"]))
    if rng.random() < family["near_target_prob"] and contact.box_min is not None:
        anchor = 0.5 * (np.asarray(contact.box_min) + np.asarray(contact.box_max))
        center = anchor + rng.uniform(*family["near_target"]) * _random_unit(rng)
    else:
        center = np.asarray(workspace_points[int(rng.integers(len(workspace_points)))], dtype=float)
        center = center + rng.uniform(0.0, family["center_offset_max"]) * _random_unit(rng)
    if rng.random() < family["bar_prob"]:
        axis = _random_unit(rng)
        spacing = 1.5 * radius
        n = family["bar_spheres"]
        return [Obstacle(center + (i - (n - 1) / 2) * spacing * axis, radius) for i in range(n)]
    return [Obstacle(center, radius)]


def sample_scene_obstacles(family, workspace_points, goals, rng, planner_rng, rollout_frames, rollout_id,
                           include_hand=True, max_candidates=300):
    """First obstacle set drawn from `family` (FAMILIES entry merged with
    FAMILY_DEFAULTS) whose every sphere keeps MIN_START_CLEARANCE from the
    start pose and clears the task goals, whose union hit rate is in the
    family's band, and that leaves a feasible path. Returns (list of
    Obstacle or None, info)."""
    family = {**FAMILY_DEFAULTS, **family}
    contact, completion = goals
    rejected = {"start_clearance": 0, "target_keepout": 0, "hit_rate": 0, "infeasible": 0}
    for candidate in range(1, max_candidates + 1):
        if "cluster_radius" in family:
            obstacles = _draw_local_cluster(family, workspace_points, goals, rng)
            if obstacles is None:
                rejected.setdefault("overlap", 0)
                rejected["overlap"] += 1
                continue
            n_groups = len(obstacles)
        else:
            n_groups = int(rng.integers(family["n"][0], family["n"][1] + 1))
            obstacles = [o for _ in range(n_groups) for o in _draw_obstacle_group(family, workspace_points, goals, rng)]
        start_clearance = arm_obstacle_clearance(START_JOINT_ANGLES, obstacles)
        if start_clearance < max(MIN_START_CLEARANCE, PLANNER_MARGIN):
            rejected["start_clearance"] += 1
            continue
        if not all(obstacle_clears_targets(o.center, o.radius, goals) for o in obstacles):
            rejected["target_keepout"] += 1
            continue
        hit_rate = rollout_hit_rate(rollout_frames, rollout_id, obstacles, include_hand)
        if not family["hit_rate_range"][0] <= hit_rate < family["hit_rate_range"][1]:
            rejected["hit_rate"] += 1
            continue
        # A larger path margin = the planner's own margin on inflated spheres.
        extra = family.get("planner_margin", PLANNER_MARGIN) - PLANNER_MARGIN
        planned = [Obstacle(o.center, o.radius + extra) for o in obstacles] if extra > 0 else obstacles
        result = check_feasibility(START_JOINT_ANGLES, contact, completion, planned, planner_rng)
        if not result["feasible"]:
            rejected["infeasible"] += 1
            continue
        return obstacles, {"candidates_drawn": candidate, "start_clearance": float(start_clearance),
                           "hit_rate": hit_rate, "n_groups": n_groups, "rejected": rejected, "planner": result}
    return None, {"candidates_drawn": max_candidates, "rejected": rejected}


def rollout_hit_rate(frames, rollout_id, obstacles, include_hand=False):
    """Fraction of the recorded rollouts (frames (n_steps, 9, 3) with each
    step's rollout_id) whose link capsules -- plus, with `include_hand`, the
    flange -> TCP capsule (GRIPPER_TIP_RADIUS, any opening) -- touch any of
    `obstacles` (one Obstacle or a list) at some step."""
    obstacles = obstacles if isinstance(obstacles, (list, tuple)) else [obstacles]
    frames = np.asarray(frames, dtype=float)
    radii = list(LINK_RADIUS)
    if include_hand:
        frames = np.concatenate([frames, fingertip_from_frames(frames)[:, None, :]], axis=1)
        radii.append(GRIPPER_TIP_RADIUS)
    a, b = frames[:, :-1, :], frames[:, 1:, :]
    ab = b - a
    ab_sq = np.maximum(np.einsum("snk,snk->sn", ab, ab), 1e-12)
    hit_step = np.zeros(len(frames), dtype=bool)
    for obstacle in obstacles:
        center = np.asarray(obstacle.center)
        t = np.clip(np.einsum("snk,snk->sn", center - a, ab) / ab_sq, 0.0, 1.0)
        dist = np.linalg.norm(a + t[..., None] * ab - center, axis=-1)
        hit_step |= np.any(dist - np.asarray(radii)[None, :] - obstacle.radius <= 0.0, axis=1)
    ids = np.unique(rollout_id)
    return float(np.mean([hit_step[rollout_id == r].any() for r in ids])) if len(ids) else 0.0


def save_manifest(path, manifest):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)


def load_scenes(path, get_env_state_for_initial_condition=None):
    """(manifest, scenes) with each scene's `obstacles` as a list of Obstacle
    (single-obstacle manifests give a one-element list) and `eval_sequence`
    as a 1-tuple. With `get_env_state_for_initial_condition`,
    also checks that CALVIN still reproduces every stored scene_obs."""
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    scenes = []
    for scene in manifest["scenes"]:
        if get_env_state_for_initial_condition is not None:
            _, scene_obs = get_env_state_for_initial_condition(scene["initial_state"])
            if not np.allclose(scene_obs, scene["scene_obs"], atol=1e-9):
                raise ValueError(f"scene {scene['scene_id']}: scene_obs no longer matches the manifest")
        if "obstacles" in scene:
            obstacles = [Obstacle(center=np.asarray(o["center_local"]), radius=o["radius"]) for o in scene["obstacles"]]
        else:
            obstacles = [Obstacle(center=np.asarray(scene["obstacle_center_local"]), radius=scene["obstacle_radius"])]
        scenes.append(dict(scene, eval_sequence=(scene["task_name"],), obstacles=obstacles))
    return manifest, scenes
