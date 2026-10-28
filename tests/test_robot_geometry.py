import numpy as np

from shortstop.calvin.robot_geometry import (
    FLANGE_FRAME_INDEX,
    FRAME_RADIUS,
    GRIPPER_TIP_OFFSET,
    GRIPPER_TIP_RADIUS,
    JOINT_LIMITS,
    LINK_RADIUS,
    N_JOINTS,
    all_frame_jacobians,
    box_to_local_frame,
    capsule_segments,
    end_effector_jacobian,
    finger_tip_capsules,
    gripper_tip_jacobian,
    gripper_tip_position,
    numerical_jacobian,
    panda_frames,
    point_to_box_distance,
    point_to_segment_distance,
    to_local_frame,
    to_world_frame,
    within_joint_limits,
)


def test_panda_frames_returns_9_positions_and_flange_is_above_base():
    q = np.zeros(N_JOINTS)
    frames = panda_frames(q)
    assert frames.shape == (9, 3)
    assert np.allclose(frames[0], [0.0, 0.0, 0.0])  # base frame at origin
    # flange should be a plausible distance from the base -- not a precise
    # spec check (see robot_geometry.py's DH-table caveat), just "not
    # degenerate" (zero, or absurdly far)
    reach = np.linalg.norm(frames[-1])
    assert 0.3 < reach < 1.5


def test_flange_frame_index_names_the_last_panda_frame():
    q = np.random.default_rng(0).uniform(-0.5, 0.5, size=N_JOINTS)
    frames = panda_frames(q)
    assert FLANGE_FRAME_INDEX == len(frames) - 1
    assert np.allclose(frames[FLANGE_FRAME_INDEX], frames[-1])


def test_frame_radius_covers_all_9_frames_and_flange_includes_the_fingertip_reach():
    assert len(FRAME_RADIUS) == 9
    # every interior frame's radius is at least as big as its two
    # adjacent links' own radii (frame sits at their junction)
    for i in range(1, len(LINK_RADIUS)):
        assert FRAME_RADIUS[i] >= LINK_RADIUS[i - 1]
        assert FRAME_RADIUS[i] >= LINK_RADIUS[i]
    # flange's radius must be big enough to cover the fingers reaching
    # past it, not just link7's own thickness
    assert FRAME_RADIUS[FLANGE_FRAME_INDEX] >= GRIPPER_TIP_OFFSET + GRIPPER_TIP_RADIUS


def test_only_upstream_joints_affect_a_given_frame():
    """Frame i's *origin* only depends on joint angles 0..i-2, one step
    earlier than one might naively guess: theta_i rotates frame i about its
    own z-axis relative to frame i-1, which does not move frame i's own
    origin (only how frame i's axes are oriented, which matters for frame
    i+1's origin) -- a basic structural property of the DH chain, and
    exactly why the Panda's joint 7 (a wrist roll whose axis passes through
    the flange) does not move the flange position, only its orientation."""
    rng = np.random.default_rng(1)
    q = rng.uniform(-0.3, 0.3, size=N_JOINTS)
    frames_before = panda_frames(q)

    q_perturbed = q.copy()
    q_perturbed[3] += 0.2  # perturb joint 4 (index 3)
    frames_after = panda_frames(q_perturbed)

    # frames 0..4 (base through joint 4's own origin) must be unchanged;
    # frame 5 onward (downstream of joint 4's rotation) must move
    assert np.allclose(frames_before[:5], frames_after[:5])
    assert not np.allclose(frames_before[5], frames_after[5])


def test_numerical_jacobian_matches_finite_difference_of_fk():
    """Self-consistency check: the Jacobian's own linear prediction of a
    small joint perturbation's effect on position should match panda_frames
    evaluated at that perturbed config."""
    rng = np.random.default_rng(2)
    q = rng.uniform(-0.4, 0.4, size=N_JOINTS)
    J = end_effector_jacobian(q)

    dq = rng.uniform(-1e-4, 1e-4, size=N_JOINTS)
    predicted = panda_frames(q)[-1] + J @ dq
    actual = panda_frames(q + dq)[-1]
    assert np.allclose(predicted, actual, atol=1e-6)


def test_numerical_jacobian_for_a_middle_sphere_frame_is_nonzero_and_shaped():
    q = np.random.default_rng(3).uniform(-0.4, 0.4, size=N_JOINTS)
    J = numerical_jacobian(q, frame_index=3)
    assert J.shape == (3, N_JOINTS)
    assert np.linalg.norm(J) > 0


def test_all_frame_jacobians_matches_perturbed_fk_for_every_frame():
    """Same self-consistency check as
    test_numerical_jacobian_matches_finite_difference_of_fk above, but
    for EVERY panda_frames() frame (0..8), not just the flange: the
    Jacobian's own linear prediction of a small joint perturbation's
    effect on position must match real forward kinematics, for every
    frame, regardless of how the Jacobian was computed. Frame 0 (base)
    must be exactly the zero matrix -- it never moves at all, regardless
    of joint angles."""
    rng = np.random.default_rng(5)
    q = rng.uniform(-0.5, 0.5, size=N_JOINTS)
    positions, jacobians = all_frame_jacobians(q)
    assert np.allclose(positions, panda_frames(q))

    assert np.allclose(jacobians[0], 0.0)  # base frame never moves

    dq = rng.uniform(-1e-4, 1e-4, size=N_JOINTS)
    perturbed_positions = panda_frames(q + dq)
    for k in range(9):
        predicted = positions[k] + jacobians[k] @ dq
        assert np.allclose(predicted, perturbed_positions[k], atol=1e-6), f"frame {k}"


def test_gripper_tip_jacobian_matches_perturbed_fk():
    """Same linear-prediction cross-check as test_all_frame_jacobians_
    matches_perturbed_fk_for_every_frame, for gripper_tip_jacobian's own
    closed-form value against gripper_tip_position (a point rigidly
    attached past the flange, not itself one of the 9 panda_frames()
    frames)."""
    rng = np.random.default_rng(6)
    q = rng.uniform(-0.5, 0.5, size=N_JOINTS)
    J = gripper_tip_jacobian(q)

    dq = rng.uniform(-1e-4, 1e-4, size=N_JOINTS)
    predicted = gripper_tip_position(q) + J @ dq
    actual = gripper_tip_position(q + dq)
    assert np.allclose(predicted, actual, atol=1e-6)


def test_point_to_box_distance_is_zero_when_point_is_inside():
    box_min, box_max = np.array([0.0, 0.0, 0.0]), np.array([1.0, 2.0, 3.0])
    assert point_to_box_distance([0.5, 1.0, 1.5], box_min, box_max) == 0.0
    assert point_to_box_distance(box_min, box_min, box_max) == 0.0  # exactly on the boundary


def test_point_to_box_distance_clamps_to_the_nearest_face():
    box_min, box_max = np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])
    # 2 units past the +x face, otherwise inside the box's y/z range
    assert np.isclose(point_to_box_distance([3.0, 0.5, 0.5], box_min, box_max), 2.0)
    # outside on 2 axes at once -- distance to the nearest CORNER
    assert np.isclose(point_to_box_distance([2.0, 2.0, 0.5], box_min, box_max), np.sqrt(2.0))


def test_point_to_box_distance_degenerate_box_is_plain_point_to_point():
    p = np.array([1.0, 2.0, 3.0])
    other = np.array([4.0, 2.0, 3.0])
    assert np.isclose(point_to_box_distance(other, p, p), 3.0)


def test_box_to_local_frame_matches_to_local_frame_under_pure_translation():
    """CALVIN's own scenes use an identity base rotation (e.g.
    calvin_scene_D.yaml's robot_base_orientation=[0,0,0]) -- under pure
    translation, transforming a box's corners and re-enclosing them must
    exactly reproduce the same box shifted by -base_position, matching
    plain to_local_frame applied to box_min/box_max directly."""
    base_position = [-0.34, -0.46, 0.24]
    base_orientation = [0.0, 0.0, 0.0, 1.0]
    box_min, box_max = np.array([0.1, 0.2, 0.3]), np.array([0.6, 0.4, 0.5])

    local_min, local_max = box_to_local_frame(box_min, box_max, base_position, base_orientation)

    assert np.allclose(local_min, to_local_frame(box_min, base_position, base_orientation))
    assert np.allclose(local_max, to_local_frame(box_max, base_position, base_orientation))


def test_box_to_local_frame_is_a_conservative_superset_under_rotation():
    """A genuinely rotated box is no longer axis-aligned -- re-enclosing
    its rotated corners in a new AABB must only ever GROW the box (safe
    direction for both g(a) and obstacle-exclusion), never shrink it
    below the true rotated shape's own extent."""
    base_position = [0.0, 0.0, 0.0]
    quarter_turn_z = [0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]
    box_min, box_max = np.array([0.0, 0.0, 0.0]), np.array([2.0, 1.0, 1.0])

    local_min, local_max = box_to_local_frame(box_min, box_max, base_position, quarter_turn_z)

    # the rotated box's own diagonal (length 1) must fit inside the new
    # enclosing box on both x and y (a 90-degree rotation swaps which
    # original axis maps to which new one, but the enclosing box must
    # still be at least as large as the rotated shape along each)
    assert (local_max - local_min)[0] >= 1.0 - 1e-9
    assert (local_max - local_min)[1] >= 1.0 - 1e-9


def test_capsule_segments_covers_every_consecutive_frame_pair():
    q = np.random.default_rng(4).uniform(-0.3, 0.3, size=N_JOINTS)
    frames = panda_frames(q)
    segments = capsule_segments(q)

    assert len(segments) == len(LINK_RADIUS) == 8
    for i, (point_a, point_b, radius) in enumerate(segments):
        assert np.allclose(point_a, frames[i])
        assert np.allclose(point_b, frames[i + 1])
        assert radius == LINK_RADIUS[i]


def test_point_to_segment_distance_clamps_to_the_nearer_endpoint():
    a, b = np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])

    # perpendicular to the middle of the segment -> exact perpendicular distance
    assert np.isclose(point_to_segment_distance([0.5, 2.0, 0.0], a, b), 2.0)
    # beyond endpoint b -> clamps to b, not the infinite line
    assert np.isclose(point_to_segment_distance([2.0, 0.0, 0.0], a, b), 1.0)
    # beyond endpoint a -> clamps to a
    assert np.isclose(point_to_segment_distance([-1.0, 0.0, 0.0], a, b), 1.0)
    # a degenerate zero-length segment falls back to point-to-point
    assert np.isclose(point_to_segment_distance([3.0, 0.0, 0.0], a, a), 3.0)


def test_gripper_tip_position_is_exactly_offset_beyond_the_flange():
    rng = np.random.default_rng(5)
    q = rng.uniform(-0.4, 0.4, size=N_JOINTS)
    flange = panda_frames(q)[-1]
    tip = gripper_tip_position(q)

    # exactly GRIPPER_TIP_OFFSET further out, regardless of joint config
    assert np.isclose(np.linalg.norm(tip - flange), GRIPPER_TIP_OFFSET)


def test_within_joint_limits_accepts_a_valid_config_and_rejects_an_out_of_range_joint():
    # a real, well-within-limits Franka "ready" pose -- NOT all-zeros:
    # joint 4 (index 3)'s own real range is entirely negative
    # ([-3.0718, -0.0698], see JOINT_LIMITS), so q=0 is itself already
    # physically invalid for that joint alone -- not a usable "valid"
    # baseline for this test.
    q_home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    assert within_joint_limits(q_home) is True

    q = q_home.copy()
    q[1] = JOINT_LIMITS[1, 1] + 0.1  # push joint 1 just past its upper bound
    assert within_joint_limits(q) is False

    q = q_home.copy()
    q[3] = JOINT_LIMITS[3, 0] - 0.1  # push joint 4 just past its lower bound
    assert within_joint_limits(q) is False


def test_to_local_frame_is_the_exact_inverse_of_to_world_frame():
    base_position = [-0.34, -0.46, 0.24]  # calvin_scene_D.yaml's real robot_base_position
    base_orientation = [0.0, 0.0, 0.0, 1.0]  # identity rotation (scene A-D's own convention)
    local_point = np.array([1.0, 2.0, 3.0])

    world_point = to_world_frame(local_point, base_position, base_orientation)
    assert np.allclose(world_point, np.array(base_position) + local_point)  # identity rotation -> pure translation
    assert np.allclose(to_local_frame(world_point, base_position, base_orientation), local_point)


def test_to_local_frame_inverts_a_real_rotation_too():
    base_position = [0.3, 0.15, 0.6]
    quarter_turn_z = [0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]  # 90 deg about z, xyzw convention
    local_point = np.array([1.0, 0.0, 0.0])

    world_point = to_world_frame(local_point, base_position, quarter_turn_z)
    assert np.allclose(to_local_frame(world_point, base_position, quarter_turn_z), local_point, atol=1e-9)


def test_finger_tip_capsules_coincide_with_gripper_tip_position_when_closed():
    q = np.random.default_rng(7).uniform(-0.3, 0.3, size=N_JOINTS)
    (left_near, left_far), (right_near, right_far) = finger_tip_capsules(q, gripper_width=0.0)
    tip = gripper_tip_position(q)
    # closed (width=0) -- both fingers collapse onto the same centerline
    # axis gripper_tip_position() itself uses.
    assert np.allclose(left_near, right_near)
    assert np.allclose(left_far, tip)
    assert np.allclose(right_far, tip)


def test_finger_tip_capsules_separate_by_the_real_gripper_width_when_open():
    q = np.random.default_rng(7).uniform(-0.3, 0.3, size=N_JOINTS)
    (left_near, left_far), (right_near, right_far) = finger_tip_capsules(q, gripper_width=0.08)
    assert np.isclose(np.linalg.norm(left_near - right_near), 0.08)
    assert np.isclose(np.linalg.norm(left_far - right_far), 0.08)


def test_gripper_tip_position_direction_tracks_flange_orientation():
    """The tip's *direction* from the flange should track the arm's own
    orientation (it's along the flange's local z-axis) -- two unrelated
    configs should generally point the tip a different way, not always
    the same fixed offset in space. (Perturbing joint 5 alone is *not* a
    reliable way to test this: at this arm's home config, joint 5's own
    rotation axis happens to coincide with the flange's final z-axis, so
    it only rolls the flange -- rotates its x/y axes -- without changing
    which way z itself points; not a bug, just not a useful probe here.)
    """
    rng = np.random.default_rng(5)
    q0 = rng.uniform(-0.4, 0.4, size=N_JOINTS)
    q1 = rng.uniform(-0.4, 0.4, size=N_JOINTS)

    offset0 = gripper_tip_position(q0) - panda_frames(q0)[-1]
    offset1 = gripper_tip_position(q1) - panda_frames(q1)[-1]
    assert not np.allclose(offset0, offset1)
