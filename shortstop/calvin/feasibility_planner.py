"""Feasibility oracle for the frozen CALVIN obstacle scenes: "with this obstacle, can the arm still reach the task
target AND carry on to the task-completion pose without touching it?"

Pass/fail only. The path it finds is thrown away -- never shown to the
policy, never a target any shield must reproduce. It runs once per
candidate scene, as the last gate before the scene is frozen, and stays
independent of every shield under comparison:
- its collision margin is its own constant (PLANNER_MARGIN), never a
  shield's w_bar/model_error/trust_region -- otherwise the pass/fail line
  would move whenever a shield is retuned;
- it is a classical planner (RRT-Connect, Kuffner & LaValle 2000 -- see
  docs/calvin/rrt_connect_citation.md) in 7-DOF joint space over
  the same analytic geometry as everything else (robot_geometry: 8 link
  capsules + the flange->fingertip capsule, JOINT_LIMITS).

What "feasible" means (2 stages, each a joint-space RRT-Connect query):
1. start config -> CONTACT: fingertip within CONTACT_TOL of the task's
   contact box, gripper pointing down within DOWN_CONE_DEG;
2. that contact config -> COMPLETION: the pose the task ends in (drawer
   pulled by its threshold, block pushed/lifted, switch flipped, wrist
   turned for rotate) -- see TaskGoal.
Goal configs come from IK (several random restarts, a redundant 7-DOF arm
has many solutions per pose) and seed the goal tree. RRT-Connect is only
probabilistically complete, so a "not found" is retried with a fresh seed
before a scene is rejected (a false reject just costs one more candidate;
a false accept would put an impossible scene in the test set).

Known simplifications (same scope as the rest of the CALVIN pipeline): no
table/furniture/self collision, only the obstacle; the completion pose is
a fingertip goal, not a simulated push/pull of the object.

All geometry here is in the robot-base-LOCAL frame (robot_geometry's own
frame); callers convert world-frame targets with box_to_local_frame.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .robot_geometry import (
    GRIPPER_TIP_OFFSET, GRIPPER_TIP_RADIUS, JOINT_LIMITS, LINK_RADIUS, N_JOINTS, _panda_transforms,
    point_to_box_distance, point_to_segment_distance,
)

# Planner's own collision margin (m) -- deliberately NOT any shield's
# w_bar/model_error, see module docstring.
PLANNER_MARGIN = 0.01

# Fingertip-to-target-box distance (m) that counts as "reached".
CONTACT_TOL = 0.03

# Gripper must point down within this cone (deg) at contact/completion --
# CALVIN's policy works top-down; a sideways reach would make the oracle
# accept poses the real task never uses.
DOWN_CONE_DEG = 45.0

# Joint-space step (rad) for growing trees, and the interpolation
# resolution (rad, max over joints) for edge collision checks.
RRT_STEP = 0.15
EDGE_RESOLUTION = 0.02

RRT_MAX_ITERS = 3000
IK_RESTARTS = 12
N_GOAL_CONFIGS = 4


def _arm_points(joint_angles):
    """(frames (9, 3), fingertip (3,), flange z-axis (3,)) from one FK pass."""
    transforms = _panda_transforms(joint_angles)
    frames = np.array([T[:3, 3] for T in transforms])
    flange = transforms[-1]
    tip = flange[:3, 3] + flange[:3, :3] @ np.array([0.0, 0.0, GRIPPER_TIP_OFFSET])
    return frames, tip, flange[:3, 2]


def arm_obstacle_clearance(joint_angles, obstacles):
    """min over (8 link capsules, flange->fingertip capsule) and obstacles of
    surface-to-surface distance; <= 0 means touching. +inf with no obstacles."""
    frames, tip, _ = _arm_points(joint_angles)
    segments = [(frames[i], frames[i + 1], LINK_RADIUS[i]) for i in range(len(LINK_RADIUS))]
    segments.append((frames[-1], tip, GRIPPER_TIP_RADIUS))
    best = np.inf
    for obstacle in obstacles:
        center = np.asarray(obstacle.center, dtype=float)
        for a, b, r in segments:
            best = min(best, point_to_segment_distance(center, a, b) - r - obstacle.radius)
    return best


def within_limits(joint_angles):
    return bool(np.all(joint_angles >= JOINT_LIMITS[:, 0]) and np.all(joint_angles <= JOINT_LIMITS[:, 1]))


def is_state_valid(joint_angles, obstacles, margin=PLANNER_MARGIN):
    return within_limits(joint_angles) and arm_obstacle_clearance(joint_angles, obstacles) >= margin


def is_edge_valid(q_a, q_b, obstacles, margin=PLANNER_MARGIN):
    """Every interpolated config from q_a to q_b (inclusive of q_b) is valid."""
    n = max(1, int(np.ceil(np.max(np.abs(q_b - q_a)) / EDGE_RESOLUTION)))
    for t in np.linspace(0.0, 1.0, n + 1)[1:]:
        if not is_state_valid(q_a + t * (q_b - q_a), obstacles, margin):
            return False
    return True


@dataclass
class TaskGoal:
    """A fingertip goal: within `tol` of box (box_min, box_max), gripper down.
    `wrist_turn` (rad, optional) instead means "rotate joint 7 by this much
    from the contact config" -- rotate tasks' completion, where the block
    turns in place."""
    box_min: np.ndarray = None
    box_max: np.ndarray = None
    tol: float = CONTACT_TOL
    wrist_turn: float = None
    meta: dict = field(default_factory=dict)


def goal_error(joint_angles, goal):
    """(fingertip distance beyond tol, down-cone violation in cos units),
    both 0 when the config satisfies `goal`."""
    _, tip, z_axis = _arm_points(joint_angles)
    dist = point_to_box_distance(tip, goal.box_min, goal.box_max)
    cos_min = np.cos(np.radians(DOWN_CONE_DEG))
    return max(0.0, dist - goal.tol), max(0.0, cos_min - float(-z_axis[2]))


def satisfies_goal(joint_angles, goal):
    dist_err, cone_err = goal_error(joint_angles, goal)
    return dist_err <= 1e-6 and cone_err <= 1e-6


def solve_ik(goal, seed_configs, obstacles, rng, n_solutions=N_GOAL_CONFIGS, restarts=IK_RESTARTS):
    """Up to `n_solutions` distinct valid configs satisfying `goal`, from
    bounded L-BFGS-B started at each of `seed_configs`, then random configs."""
    lower, upper = JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1]
    cos_min = np.cos(np.radians(DOWN_CONE_DEG))

    def cost(q):
        _, tip, z_axis = _arm_points(q)
        dist = point_to_box_distance(tip, goal.box_min, goal.box_max)
        return max(0.0, dist - 0.5 * goal.tol) ** 2 + 0.1 * max(0.0, cos_min + 0.05 - float(-z_axis[2])) ** 2

    starts = [np.asarray(q, dtype=float) for q in seed_configs]
    starts += [rng.uniform(lower, upper) for _ in range(restarts)]
    solutions = []
    for q0 in starts:
        result = minimize(cost, np.clip(q0, lower, upper), method="L-BFGS-B", bounds=list(zip(lower, upper)),
                          options={"maxiter": 200})
        q = result.x
        if not satisfies_goal(q, goal) or not is_state_valid(q, obstacles):
            continue
        if any(np.max(np.abs(q - s)) < 0.1 for s in solutions):
            continue
        solutions.append(q)
        if len(solutions) >= n_solutions:
            break
    return solutions


def _nearest(tree, q):
    nodes = np.asarray(tree["nodes"])
    return int(np.argmin(np.max(np.abs(nodes - q), axis=1)))


def _extend(tree, q_target, obstacles):
    """One RRT step from the nearest node toward q_target. Returns
    ("reached"|"advanced"|"trapped", new node index or None)."""
    i = _nearest(tree, q_target)
    q_near = tree["nodes"][i]
    delta = q_target - q_near
    dist = np.max(np.abs(delta))
    if dist <= RRT_STEP:
        q_new, status = q_target, "reached"
    else:
        q_new, status = q_near + delta * (RRT_STEP / dist), "advanced"
    if not is_edge_valid(q_near, q_new, obstacles):
        return "trapped", None
    tree["nodes"].append(q_new)
    tree["parent"].append(i)
    return status, len(tree["nodes"]) - 1


def _connect(tree, q_target, obstacles):
    while True:
        status, idx = _extend(tree, q_target, obstacles)
        if status != "advanced":
            return status, idx


def _path_to_root(tree, idx):
    path = []
    while idx is not None and idx >= 0:
        path.append(tree["nodes"][idx])
        idx = tree["parent"][idx]
    return path[::-1]


def rrt_connect(q_start, goal_configs, obstacles, rng, max_iters=RRT_MAX_ITERS):
    """Joint-space RRT-Connect from q_start to ANY of goal_configs (all seed
    the goal tree). Returns (path list of configs, iterations) or (None, iters)."""
    q_start = np.asarray(q_start, dtype=float)
    if not goal_configs or not is_state_valid(q_start, obstacles):
        return None, 0
    for g in goal_configs:
        if is_edge_valid(q_start, g, obstacles):
            return [q_start, np.asarray(g, dtype=float)], 0
    start_tree = {"nodes": [q_start], "parent": [-1]}
    goal_tree = {"nodes": [np.asarray(g, dtype=float) for g in goal_configs], "parent": [-1] * len(goal_configs)}
    lower, upper = JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1]
    trees = [start_tree, goal_tree]
    for it in range(1, max_iters + 1):
        a, b = trees
        q_rand = rng.uniform(lower, upper)
        status, idx = _extend(a, q_rand, obstacles)
        if status != "trapped":
            status_b, idx_b = _connect(b, a["nodes"][idx], obstacles)
            if status_b == "reached":
                path_a, path_b = _path_to_root(a, idx), _path_to_root(b, idx_b)
                path = path_a + path_b[::-1][1:] if a is start_tree else path_b + path_a[::-1][1:]
                return path, it
        trees = [b, a]
    return None, max_iters


def _completion_configs(contact_q, completion_goal, obstacles, rng):
    if completion_goal.wrist_turn is not None:
        q = np.array(contact_q, dtype=float)
        q[N_JOINTS - 1] += completion_goal.wrist_turn
        return [q] if is_state_valid(q, obstacles) else []
    return solve_ik(completion_goal, [contact_q], obstacles, rng)


def check_feasibility(q_start, contact_goal, completion_goal, obstacles, rng, attempts=2):
    """Is there a collision-free path start -> contact -> completion?
    Returns a dict: feasible (bool), reason, and per-attempt stats. `rng` is
    a numpy Generator; every attempt draws fresh IK restarts and samples."""
    q_start = np.asarray(q_start, dtype=float)
    stats = {"feasible": False, "reason": None, "attempts": []}
    if not is_state_valid(q_start, obstacles):
        stats["reason"] = "start_in_collision"
        return stats
    for _ in range(attempts):
        record = {}
        stats["attempts"].append(record)
        contacts = solve_ik(contact_goal, [q_start], obstacles, rng)
        record["n_contact_ik"] = len(contacts)
        if not contacts:
            stats["reason"] = "no_contact_ik"
            continue
        path, iters = rrt_connect(q_start, contacts, obstacles, rng)
        record["contact_iters"] = iters
        if path is None:
            stats["reason"] = "no_path_to_contact"
            continue
        completions = _completion_configs(path[-1], completion_goal, obstacles, rng)
        record["n_completion_ik"] = len(completions)
        if not completions:
            stats["reason"] = "no_completion_ik"
            continue
        path2, iters2 = rrt_connect(path[-1], completions, obstacles, rng)
        record["completion_iters"] = iters2
        if path2 is None:
            stats["reason"] = "no_path_to_completion"
            continue
        stats.update(feasible=True, reason=None, contact_config=path[-1].tolist(),
                     completion_config=path2[-1].tolist())
        return stats
    return stats
