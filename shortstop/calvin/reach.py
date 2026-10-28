"""Reach step: a capsule reachtube of the Panda arm over an action chunk.

CALVIN actions are relative TCP moves: CALVIN adds `action[:3] *
CALVIN_ACTION_SCALE` (max_rel_pos 0.02 x magic_scaling_factor_pos 1, read
from the env config the dataset ships and checked in sim) to its TCP target
every step. The prediction turns each move into a joint step through the
pseudo-inverse of the TCP Jacobian (resolved-rate IK), then places every
arm frame and link by forward kinematics.

That linearization is the model error: the predicted arm is not exactly
where CALVIN's IK puts it, and the error grows along the chunk. The pose
model (`step_joint_config` with the TCP orientation) also follows the
rotation columns, as CALVIN's IK does. The tube
covers it by inflating step k's capsules by `model_error_k + w_bar`
(model_error per step, calibrated on k-step residuals,
scripts/calibrate_arm_model_error.py; see config.MODEL_ERROR). The gripper column is not propagated.

Each tube step holds the 9 frames (base..flange) as point capsules, the 8
links as segment capsules and a "fingertip" capsule from the flange to the
TCP (fixed worst-case-open radius, GRIPPER_TIP_RADIUS). Distances to the
spherical obstacles are exact capsule-sphere distances.
"""
import numpy as np

from .config import MODEL_ERROR
from .robot_geometry import (
    FRAME_RADIUS, GRIPPER_TIP_RADIUS, LINK_RADIUS, closest_point_on_segment, gripper_tip_jacobian,
    gripper_tip_position, panda_frames,
)

# Meters of TCP motion per unit of action (see module docstring).
CALVIN_ACTION_SCALE = 0.02
# Radians per unit of the rotation columns (max_rel_orn 0.05 x magic_scaling_factor_orn 1, read
# from the env's robot at calibration time).
CALVIN_ORIENTATION_SCALE = 0.05


class Capsule:
    """Segment (a, b) with a radius that already includes every inflation;
    a == b is a point."""
    __slots__ = ("a", "b", "radius")

    def __init__(self, a, b, radius):
        self.a = np.asarray(a, dtype=float)
        self.b = np.asarray(b, dtype=float)
        self.radius = radius

    def center(self):
        return (self.a + self.b) / 2.0

    def closest_point(self, p):
        return closest_point_on_segment(p, self.a, self.b)


def _signed_distance(capsule, obstacle):
    """Capsule-to-sphere surface distance (negative = overlap)."""
    closest = capsule.closest_point(obstacle.center)
    return float(np.linalg.norm(closest - obstacle.center) - capsule.radius - obstacle.radius)


def _step_joint_config(joint_angles, task_space_delta_pos):
    """One predicted step: move the TCP by the action's position delta via
    the TCP Jacobian's pseudo-inverse."""
    J = gripper_tip_jacobian(joint_angles)
    dq = np.linalg.pinv(J) @ (task_space_delta_pos * CALVIN_ACTION_SCALE)
    return joint_angles + dq


def _rotation_from_euler(euler):
    """PyBullet's Euler convention (getQuaternionFromEuler): R = Rz Ry Rx."""
    r, p, y = euler
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _rotation_vector(R):
    """Axis-angle vector of rotation matrix R."""
    angle = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * np.sin(angle))
    return axis * angle


# Damping of the pose model's least-squares inverse: near wrist singularities
# a plain pseudo-inverse turns the rotation command into huge joint moves.
POSE_DLS_DAMPING = 0.03


def _pose_inverse(joint_angles):
    """(7, 6) damped least-squares inverse of [J_pos; J_rot] at `joint_angles`."""
    from .robot_geometry import _joint_axes_and_points, _panda_transforms
    q = np.asarray(joint_angles, dtype=float)
    axes, _ = _joint_axes_and_points(_panda_transforms(q))
    J = np.vstack([gripper_tip_jacobian(q), axes.T])
    return J.T @ np.linalg.inv(J @ J.T + POSE_DLS_DAMPING ** 2 * np.eye(6))


def pose_step_parts(joint_angles, tcp_euler, rotation_row):
    """Linearized pose model of one CALVIN step at `joint_angles`: CALVIN adds
    action[:3] * CALVIN_ACTION_SCALE to its TCP position target and
    action[3:6] * CALVIN_ORIENTATION_SCALE to its Euler orientation target.
    Returns (M, dq_rot, next_euler): dq = M @ (position move in m) + dq_rot,
    from the damped inverse of [J_pos; J_rot] with the rotation between
    consecutive targets."""
    euler = np.asarray(tcp_euler, dtype=float)
    next_euler = euler + np.asarray(rotation_row, dtype=float) * CALVIN_ORIENTATION_SCALE
    omega = _rotation_vector(_rotation_from_euler(next_euler) @ _rotation_from_euler(euler).T)
    J_inv = _pose_inverse(joint_angles)
    return J_inv[:, :3], J_inv[:, 3:] @ omega, next_euler


def position_joint_map(joint_angles, pose_model):
    """(7, 3) joint move per meter of TCP position move: pinv of the pose
    Jacobian's position columns (orientation held), or of the TCP Jacobian."""
    q = np.asarray(joint_angles, dtype=float)
    if not pose_model:
        return np.linalg.pinv(gripper_tip_jacobian(q))
    return _pose_inverse(q)[:, :3]


def step_joint_config(joint_angles, action_row, tcp_euler=None):
    """One predicted step -> (next joints, next TCP Euler or None). With the
    TCP orientation, the pose model (position + orientation columns);
    without it, the position-only model (`_step_joint_config`)."""
    q = np.asarray(joint_angles, dtype=float)
    row = np.asarray(action_row, dtype=float)
    if tcp_euler is None:
        return _step_joint_config(q, row[:3]), None
    M, dq_rot, next_euler = pose_step_parts(q, tcp_euler, row[3:6])
    return q + M @ (row[:3] * CALVIN_ACTION_SCALE) + dq_rot, next_euler


def step_prediction_residual(joint_angles, action, next_joint_angles):
    """Largest distance, over every arm frame and the TCP, between where one
    real step put the arm (`next_joint_angles`) and where `_step_joint_config`
    predicts it from `joint_angles` and `action` (first 3 columns used).
    config.MODEL_ERROR = p99 of this over real rollouts x 1.25."""
    action = np.asarray(action, dtype=float)
    predicted = _step_joint_config(np.asarray(joint_angles, dtype=float), action[:3])
    next_joint_angles = np.asarray(next_joint_angles, dtype=float)
    predicted_frames = np.vstack([panda_frames(predicted), gripper_tip_position(predicted)])
    real_frames = np.vstack([panda_frames(next_joint_angles), gripper_tip_position(next_joint_angles)])
    return float(np.max(np.linalg.norm(real_frames - predicted_frames, axis=1)))


def nominal_joint_trajectory(joint_angles, task_chunk, tcp_euler=None):
    """Predicted joint configs [q0, q1, ..., qH] for a chunk (pose model when
    the TCP orientation is given, see `step_joint_config`)."""
    q = np.asarray(joint_angles, dtype=float).copy()
    euler = tcp_euler
    trajectory = [q.copy()]
    for row in np.asarray(task_chunk, dtype=float):
        q, euler = step_joint_config(q, row, euler)
        trajectory.append(q.copy())
    return trajectory


def model_error_at(model_error, k):
    """Tube radius at step k >= 1: `model_error` is one radius for every step
    or a per-step sequence (the last value repeats beyond its length)."""
    if np.ndim(model_error) == 0:
        return float(model_error)
    return float(model_error[min(k, len(model_error)) - 1])


def propagate_arm_tube(joint_angles, task_chunk, w_bar, model_error=MODEL_ERROR, tcp_euler=None):
    """Reachtube over `task_chunk` along `nominal_joint_trajectory` (pose
    model when `tcp_euler`, the current TCP orientation, is given).

    tube[0] holds the current frames as zero-radius points; tube[k], k=1..H,
    maps frame index (0..8) -> point capsule, ("link", i) (0..7) -> link
    capsule and "fingertip" -> flange-to-TCP capsule. Radii are
    model_error_k + w_bar + the primitive's physical radius (the flange point
    uses its link radius; the fingertip capsule carries the gripper reach)."""
    q = np.asarray(joint_angles, dtype=float)
    tube = [{i: Capsule(c, c, 0.0) for i, c in enumerate(panda_frames(q))}]
    for k, q_k in enumerate(nominal_joint_trajectory(q, task_chunk, tcp_euler)[1:], 1):
        tube.append(arm_step_capsules(q_k, model_error_at(model_error, k) + w_bar))
    return tube


def arm_step_capsules(joint_angles, inflation=0.0):
    """One tube step's primitives at `joint_angles`, every radius grown by
    `inflation` (0 = the arm's own geometry as the shields model it)."""
    frames = panda_frames(joint_angles)
    capsules = {
        i: Capsule(c, c, inflation + (LINK_RADIUS[-1] if i == len(frames) - 1 else FRAME_RADIUS[i]))
        for i, c in enumerate(frames)
    }
    capsules.update({
        ("link", i): Capsule(frames[i], frames[i + 1], inflation + LINK_RADIUS[i])
        for i in range(len(LINK_RADIUS))
    })
    capsules["fingertip"] = Capsule(frames[-1], gripper_tip_position(joint_angles), inflation + GRIPPER_TIP_RADIUS)
    return capsules


def arm_step_robustness(frames_at_step, obstacles):
    """Smallest capsule-obstacle distance over one tube step; +inf without
    obstacles."""
    return min((_signed_distance(capsule, o) for capsule in frames_at_step.values() for o in obstacles),
               default=float("inf"))


def arm_robustness_to_go(tube, obstacles):
    """Paper formula (2): smallest distance over steps 1..H and all capsules."""
    return min(arm_step_robustness(frames, obstacles) for frames in tube[1:])


def arm_find_counterexample(tube, obstacles):
    """Paper formula (3): the worst (step, primitive, obstacle), as
    {step, frame (int, ("link", i) or "fingertip"), obstacle, witness (closest
    point on the capsule axis), robustness}."""
    best = None
    for k in range(1, len(tube)):
        for frame_index, capsule in tube[k].items():
            for obstacle in obstacles:
                value = _signed_distance(capsule, obstacle)
                if best is None or value < best["robustness"]:
                    best = {
                        "step": k,
                        "frame": frame_index,
                        "obstacle": obstacle,
                        "witness": capsule.closest_point(obstacle.center),
                        "robustness": value,
                    }
    return best
