"""Franka Emika Panda kinematics and collision geometry, in the robot-base
local frame (to_world_frame / to_local_frame convert).

- panda_frames: modified-DH (Craig) forward kinematics, 9 points (base, 7
  joints, flange); matches CALVIN's PyBullet robot to < 1e-5 m
  (scripts/verify_robot_geometry_against_pybullet.py; PyBullet's link state
  is the center of mass, the script reads the URDF link frame instead).
- capsule_segments: one capsule per link (radii from the collision meshes).
- gripper_tip_position: the TCP (CALVIN's tcp_link), 0.14 m past the flange.
- finger_tip_capsules: the two fingers at the real gripper width.
- Closed-form Jacobians of every frame and of the TCP.
"""
import itertools

import numpy as np

# (alpha_{i-1}, a_{i-1}, d_i) for i = 1..7, modified DH (theta_i = joint angle q_i)
PANDA_DH = [
    (0.0, 0.0, 0.333),
    (-np.pi / 2, 0.0, 0.0),
    (np.pi / 2, 0.0, 0.316),
    (np.pi / 2, 0.0825, 0.0),
    (-np.pi / 2, -0.0825, 0.384),
    (np.pi / 2, 0.0, 0.0),
    (np.pi / 2, 0.088, 0.0),
]
FLANGE_OFFSET = (0.0, 0.0, 0.107)  # (alpha, a, d) from joint 7 frame to flange, theta=0

N_JOINTS = 7

# Index of the flange in panda_frames().
FLANGE_FRAME_INDEX = N_JOINTS + 1

# Soft joint limits (rad) CALVIN's robot enforces (calvin_env conf/robot/panda.yaml).
JOINT_LIMITS = np.array([
    (-2.8973, 2.8973),
    (-1.7628, 1.7628),
    (-2.8973, 2.8973),
    (-3.0718, -0.0698),
    (-2.8973, 2.8973),
    (-0.0175, 3.7525),
    (-2.8973, 2.8973),
])

# Capsule radius of link0..link7 (frame i to frame i+1): max distance of the
# link's collision-mesh vertices (calvin_env franka_panda meshes) from the
# mesh's principal axis, rounded up to the cm.
LINK_RADIUS = [0.13, 0.11, 0.10, 0.09, 0.09, 0.10, 0.11, 0.06]

# Flange -> TCP distance along the flange z axis. CALVIN loads
# panda_longer_finger.urdf, whose tcp_joint is 0.14 (0.1 in plain panda.urdf);
# checked against PyBullet (scripts/verify_gripper_geometry_against_pybullet.py).
GRIPPER_TIP_OFFSET = 0.14

# Radius of the reachtube's flange->TCP capsule, fixed for any gripper
# opening: 0.04 max finger travel per side + 0.02 finger thickness.
GRIPPER_TIP_RADIUS = 0.06

# Finger geometry from the URDF: the hand is the flange rotated by
# HAND_YAW_OFFSET about z; both fingers start FINGER_JOINT_Z_OFFSET down the
# hand axis and slide +-y by half the gripper width; FINGER_RADIUS is the
# finger mesh thickness (scripts/verify_robot_geometry_mesh_constants.py).
HAND_YAW_OFFSET = -0.785398163397
FINGER_JOINT_Z_OFFSET = 0.0584
FINGER_RADIUS = 0.02

# Radius of each frame point: the thicker of its two adjacent links. The
# flange entry folds in the whole gripper reach; the reachtube and the QP
# shields use the flange's link radius plus a flange->TCP capsule instead.
FRAME_RADIUS = (
    [LINK_RADIUS[0]]
    + [max(LINK_RADIUS[i - 1], LINK_RADIUS[i]) for i in range(1, len(LINK_RADIUS))]
    + [GRIPPER_TIP_OFFSET + GRIPPER_TIP_RADIUS]
)


def _mdh_transform(alpha, a, d, theta):
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct, -st, 0.0, a],
        [st * ca, ct * ca, -sa, -sa * d],
        [st * sa, ct * sa, ca, ca * d],
        [0.0, 0.0, 0.0, 1.0],
    ])


def _panda_transforms(joint_angles):
    """The 9 frames of panda_frames() as 4x4 transforms."""
    joint_angles = np.asarray(joint_angles, dtype=float)
    if joint_angles.shape != (N_JOINTS,):
        raise ValueError(f"expected {N_JOINTS} joint angles, got shape {joint_angles.shape}")

    T = np.eye(4)
    transforms = [T.copy()]
    for (alpha, a, d), theta in zip(PANDA_DH, joint_angles):
        T = T @ _mdh_transform(alpha, a, d, theta)
        transforms.append(T.copy())
    T = T @ _mdh_transform(*FLANGE_OFFSET, 0.0)
    transforms.append(T.copy())
    return transforms


def panda_frames(joint_angles):
    """(9, 3) positions of base, joints 1-7 and flange for 7 joint angles."""
    return np.array([T[:3, 3] for T in _panda_transforms(joint_angles)])


def within_joint_limits(joint_angles):
    """Are all joint angles within JOINT_LIMITS?"""
    joint_angles = np.asarray(joint_angles, dtype=float)
    return bool(np.all(joint_angles >= JOINT_LIMITS[:, 0]) and np.all(joint_angles <= JOINT_LIMITS[:, 1]))


def gripper_tip_position(joint_angles):
    """The TCP: GRIPPER_TIP_OFFSET along the flange's z axis."""
    flange_T = _panda_transforms(joint_angles)[-1]
    flange_position = flange_T[:3, 3]
    flange_rotation = flange_T[:3, :3]
    return flange_position + flange_rotation @ np.array([0.0, 0.0, GRIPPER_TIP_OFFSET])


def finger_tip_capsules(joint_angles, gripper_width):
    """((left_near, left_far), (right_near, right_far)) finger segments for
    `gripper_width` = CALVIN's gripper_opening_width (sum of both finger
    joints, 0 = closed); each finger sits gripper_width / 2 off the axis."""
    flange_T = _panda_transforms(joint_angles)[-1]
    flange_position = flange_T[:3, 3]
    flange_rotation = flange_T[:3, :3]

    ca, sa = np.cos(HAND_YAW_OFFSET), np.sin(HAND_YAW_OFFSET)
    hand_yaw = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
    hand_rotation = flange_rotation @ hand_yaw

    half_width = gripper_width / 2.0
    capsules = []
    for side in (1.0, -1.0):
        near = flange_position + hand_rotation @ np.array([0.0, side * half_width, FINGER_JOINT_Z_OFFSET])
        far = flange_position + hand_rotation @ np.array([0.0, side * half_width, GRIPPER_TIP_OFFSET])
        capsules.append((near, far))
    return capsules[0], capsules[1]


def capsule_segments(joint_angles):
    """[(frame i, frame i+1, LINK_RADIUS[i])] for the 8 links."""
    frames = panda_frames(joint_angles)
    return [(frames[i], frames[i + 1], LINK_RADIUS[i]) for i in range(len(LINK_RADIUS))]


def closest_point_on_segment(point, a, b):
    """Point of segment ab closest to `point` (a if a == b)."""
    point = np.asarray(point, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ab = b - a
    length_sq = float(ab @ ab)
    if length_sq < 1e-12:
        return a.copy()
    t = np.clip(float((point - a) @ ab) / length_sq, 0.0, 1.0)
    return a + t * ab


def point_to_segment_distance(point, a, b):
    """Distance from `point` to segment ab (no radius subtracted)."""
    point = np.asarray(point, dtype=float)
    closest = closest_point_on_segment(point, a, b)
    return float(np.linalg.norm(point - closest))


def point_to_box_distance(point, box_min, box_max):
    """Distance from `point` to the axis-aligned box (0 inside); a point is a
    degenerate box."""
    point = np.asarray(point, dtype=float)
    box_min = np.asarray(box_min, dtype=float)
    box_max = np.asarray(box_max, dtype=float)
    clamped = np.clip(point, box_min, box_max)
    return float(np.linalg.norm(point - clamped))


def _joint_axes_and_points(transforms):
    """Each revolute joint's axis direction and a point on it, in the base
    frame: joint i rotates about the z axis of transforms[i-1] @
    Rot_x(alpha) Trans_x(a), the modified-DH frame before its theta."""
    axes = np.zeros((N_JOINTS, 3))
    points = np.zeros((N_JOINTS, 3))
    for idx in range(N_JOINTS):
        alpha, a, _d = PANDA_DH[idx]
        partial = transforms[idx] @ _mdh_transform(alpha, a, 0.0, 0.0)
        axes[idx] = partial[:3, 2]
        points[idx] = partial[:3, 3]
    return axes, points


def all_frame_jacobians(joint_angles):
    """(positions (9, 3), Jacobians (9, 3, 7)) of every panda_frames() point,
    closed form: column i of frame k is z_i x (p_k - p_i) for joints i <= k,
    0 for later joints."""
    transforms = _panda_transforms(joint_angles)
    positions = np.array([T[:3, 3] for T in transforms])
    axes, axis_points = _joint_axes_and_points(transforms)

    jacobians = np.zeros((len(transforms), 3, N_JOINTS))
    for k in range(len(transforms)):
        n_affecting = min(k, N_JOINTS)
        for idx in range(n_affecting):
            jacobians[k, :, idx] = np.cross(axes[idx], positions[k] - axis_points[idx])
    return positions, jacobians


def numerical_jacobian(joint_angles, frame_index):
    """(3, 7) Jacobian of frame `frame_index` (closed form despite the name)."""
    _positions, jacobians = all_frame_jacobians(joint_angles)
    return jacobians[frame_index]


def end_effector_jacobian(joint_angles):
    """(3, 7) Jacobian of the flange."""
    return numerical_jacobian(joint_angles, frame_index=len(PANDA_DH) + 1)


def gripper_tip_jacobian(joint_angles):
    """(3, 7) Jacobian of the TCP, closed form."""
    transforms = _panda_transforms(joint_angles)
    axes, axis_points = _joint_axes_and_points(transforms)
    flange_T = transforms[-1]
    tip = flange_T[:3, 3] + flange_T[:3, :3] @ np.array([0.0, 0.0, GRIPPER_TIP_OFFSET])
    J = np.zeros((3, N_JOINTS))
    for idx in range(N_JOINTS):
        J[:, idx] = np.cross(axes[idx], tip - axis_points[idx])
    return J


def quaternion_to_rotation_matrix(quaternion):
    """pybullet xyzw quaternion -> 3x3 rotation matrix."""
    x, y, z, w = quaternion
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def to_world_frame(local_point, base_position, base_orientation):
    """Robot-base local point -> world (CALVIN's base sits at (-0.34, -0.46, 0.24))."""
    rotation = quaternion_to_rotation_matrix(base_orientation)
    return np.asarray(base_position) + rotation @ np.asarray(local_point)


def to_local_frame(world_point, base_position, base_orientation):
    """World point -> robot-base local frame."""
    rotation = quaternion_to_rotation_matrix(base_orientation)
    return rotation.T @ (np.asarray(world_point) - np.asarray(base_position))


def box_to_local_frame(box_min, box_max, base_position, base_orientation):
    """Axis-aligned world box -> axis-aligned local box enclosing its 8
    corners (exact for CALVIN's unrotated base)."""
    box_min = np.asarray(box_min, dtype=float)
    box_max = np.asarray(box_max, dtype=float)
    corners = np.array(list(itertools.product(*zip(box_min, box_max))))
    local_corners = np.array([to_local_frame(c, base_position, base_orientation) for c in corners])
    return local_corners.min(axis=0), local_corners.max(axis=0)
