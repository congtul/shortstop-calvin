import numpy as np

from shortstop.calvin.reach import (
    _step_joint_config,
    arm_find_counterexample,
    arm_robustness_to_go,
    arm_step_robustness,
    propagate_arm_tube,
    step_prediction_residual,
)
from shortstop.env import Obstacle
from shortstop.calvin.robot_geometry import (
    FLANGE_FRAME_INDEX, FRAME_RADIUS, GRIPPER_TIP_RADIUS, LINK_RADIUS, N_JOINTS, gripper_tip_position,
    panda_frames,
)


def _zero_chunk(horizon=4, action_dim=7):
    return np.zeros((horizon, action_dim))


def test_step_prediction_residual_is_zero_when_the_real_next_config_matches_the_prediction():
    q = np.random.default_rng(6).uniform(-0.3, 0.3, size=N_JOINTS)
    action = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    predicted_q = _step_joint_config(q, action[:3])
    assert np.isclose(step_prediction_residual(q, action, predicted_q), 0.0, atol=1e-9)


def test_step_prediction_residual_is_positive_when_the_real_next_config_is_perturbed():
    q = np.random.default_rng(6).uniform(-0.3, 0.3, size=N_JOINTS)
    action = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    predicted_q = _step_joint_config(q, action[:3])
    perturbed_q = predicted_q.copy()
    perturbed_q[2] += 0.05
    assert step_prediction_residual(q, action, perturbed_q) > 0.0


def test_propagate_arm_tube_zero_action_keeps_frames_at_current_position():
    q = np.random.default_rng(0).uniform(-0.3, 0.3, size=N_JOINTS)
    tube = propagate_arm_tube(q, _zero_chunk(), w_bar=0.0, model_error=0.0)
    frames = panda_frames(q)

    assert len(tube) == 5  # horizon 4 + the initial (realized) entry
    for step in tube[1:]:
        # the whole chain (9 frame capsules + 8 link capsules + 1 fingertip
        # capsule), not a coarser subset
        assert set(step.keys()) == set(range(9)) | {("link", i) for i in range(8)} | {"fingertip"}
        for i in range(9):
            capsule = step[i]
            # zero disturbance/model_error -> the capsule's own point
            # doesn't move, but its radius is still the frame's own
            # physical radius (see propagate_arm_tube's docstring), not 0.
            assert np.allclose(capsule.a, capsule.b)  # a point capsule
            # frame 8 (flange) is the one exception -- LINK_RADIUS[-1], not
            # FRAME_RADIUS[8] (which bakes in fingertip-reach inflation
            # that's now the dedicated "fingertip" capsule's own job, see
            # propagate_arm_tube's docstring).
            expected_radius = LINK_RADIUS[-1] if i == 8 else FRAME_RADIUS[i]
            assert np.isclose(capsule.radius, expected_radius, atol=1e-9)
        for i in range(8):
            capsule = step[("link", i)]
            # link capsule spans exactly the two endpoint frames, radius
            # exactly LINK_RADIUS[i] (zero disturbance/model_error here)
            assert np.allclose(capsule.a, frames[i], atol=1e-9)
            assert np.allclose(capsule.b, frames[i + 1], atol=1e-9)
            assert np.isclose(capsule.radius, LINK_RADIUS[i], atol=1e-9)
        # fingertip capsule spans flange -> TCP, radius exactly
        # GRIPPER_TIP_RADIUS (zero disturbance/model_error here)
        fingertip = step["fingertip"]
        assert np.allclose(fingertip.a, frames[-1], atol=1e-9)
        assert np.allclose(fingertip.b, gripper_tip_position(q), atol=1e-9)
        assert np.isclose(fingertip.radius, GRIPPER_TIP_RADIUS, atol=1e-9)


def test_propagate_arm_tube_inflates_with_disturbance_and_model_error():
    q = np.zeros(N_JOINTS)
    tube = propagate_arm_tube(q, _zero_chunk(horizon=1), w_bar=0.05, model_error=0.02)
    capsule = tube[1][0]
    expected_r = 0.05 + 0.02 + FRAME_RADIUS[0]  # disturbance + model_error + own radius
    assert np.isclose(capsule.radius, expected_r, atol=1e-9)


def test_arm_robustness_to_go_matches_manual_min_over_frames_and_obstacles():
    q = np.zeros(N_JOINTS)
    tube = propagate_arm_tube(q, _zero_chunk(horizon=2), w_bar=0.0, model_error=0.0)

    # obstacle placed exactly at the flange's step-1 position -> should be
    # the binding (most negative) term
    flange_pos = tube[1][FLANGE_FRAME_INDEX].center()
    obstacles = [Obstacle(center=flange_pos, radius=0.05)]

    value = arm_robustness_to_go(tube, obstacles)
    # Exact capsule-vs-sphere distance: both surfaces (flange's own
    # physical radius AND the obstacle's) are subtracted, not just the
    # obstacle's.
    # The flange's own point-capsule uses LINK_RADIUS[-1] (0.06m), not
    # FRAME_RADIUS[FLANGE_FRAME_INDEX] (0.20m, which folds in fingertip
    # reach -- that inflation is the dedicated "fingertip" capsule's job
    # instead, see propagate_arm_tube's docstring). The fingertip capsule
    # ties at the same value here (obstacle sits exactly at its own
    # "flange" end), so the binding term is unchanged either way.
    assert np.isclose(value, -(LINK_RADIUS[-1] + 0.05), atol=1e-6)


def test_link_box_catches_a_mid_link_collision_the_old_frame_only_check_would_miss():
    """Verifies link-capsule collision checking: an obstacle placed exactly
    at the MIDPOINT of a link (far from both its endpoint frames) is only
    caught because propagate_arm_tube models each link as its own capsule,
    not just the 9 per-frame point-boxes -- a frame-only check would miss
    a mid-link collision even though the arm's own volume genuinely passes
    through it. shortstop.calvin.experiment._clearance (ground truth,
    uses the real capsule chain) does catch it.
    An unfolded (non-home) config is used deliberately: at q=0 the arm
    folds back on itself, so unrelated frames' own heavily-inflated boxes
    (e.g. the flange's, FRAME_RADIUS=0.16) can coincidentally overlap a
    point that's nowhere near them in the kinematic chain -- a config
    where every frame's own distance is unambiguously positive isolates
    the link-capsule behavior being tested here."""
    q = np.array([0.0, 0.5, 0.0, -1.0, 0.0, 1.0, 0.0])
    chunk = _zero_chunk(horizon=1)
    tube = propagate_arm_tube(q, chunk, w_bar=0.0, model_error=0.0)
    frames = panda_frames(q)

    link_i = 4
    midpoint = (frames[link_i] + frames[link_i + 1]) / 2.0
    obstacles = [Obstacle(center=midpoint, radius=0.01)]

    # Using only the 9 frame-only boxes (no link/segment modeling), every
    # one stays clear (positive robustness): a frame-only check would call
    # this candidate admissible.
    frame_only_step = {k: v for k, v in tube[1].items() if isinstance(k, int)}
    old_robustness = arm_step_robustness(frame_only_step, obstacles)
    assert old_robustness > 0.0

    # arm_robustness_to_go (frames + links) correctly flags the violation,
    # via link 4's own box.
    new_robustness = arm_robustness_to_go(tube, obstacles)
    assert new_robustness < 0.0

    ce = arm_find_counterexample(tube, obstacles)
    assert ce["frame"] == ("link", link_i)


def test_arm_find_counterexample_identifies_the_violating_step_and_robustness():
    q = np.zeros(N_JOINTS)
    # nonzero action so the flange actually moves step to step -- with a
    # zero action every step is a no-op and the "obstacle at step 2's
    # position" would coincide with step 1's position too, since the
    # configuration never changes. Per-step displacement (0.3) is
    # deliberately well beyond any frame's own radius (<=0.16, see
    # robot_geometry.FRAME_RADIUS) so step 1 and step 2 stay clearly
    # separated even after propagate_arm_tube's box is inflated by each
    # frame's physical radius -- a small displacement comparable to the
    # inflation amount can make the (axis-aligned-box) closest-point
    # computation degenerate and unable to tell adjacent steps apart.
    chunk = np.tile([0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], (2, 1))
    tube = propagate_arm_tube(q, chunk, w_bar=0.0, model_error=0.0)
    flange_pos = tube[2][FLANGE_FRAME_INDEX].center()
    obstacles = [Obstacle(center=flange_pos, radius=0.05)]

    ce = arm_find_counterexample(tube, obstacles)
    assert ce["step"] == 2
    # Exact capsule-vs-sphere distance: frame 8 (flange, exactly at the
    # obstacle) is unambiguously the worst violator, since the capsule
    # model has no box approximation to create a degenerate tie with
    # frame 7. The "fingertip" capsule ties at the exact same robustness
    # value here (obstacle sits exactly at its own flange endpoint too),
    # but frame 8 was inserted into the dict first and
    # arm_find_counterexample only replaces the best-so-far on a STRICT
    # improvement, so the tie keeps frame 8 as the reported violator, not
    # "fingertip".
    assert ce["frame"] == FLANGE_FRAME_INDEX
    assert np.isclose(ce["robustness"], -(LINK_RADIUS[-1] + 0.05), atol=1e-6)


def test_pose_model_step_moves_the_tcp_and_rotates_the_hand_as_commanded():
    """First order: the TCP moves by action[:3] * 0.02 m and the hand turns
    by the rotation between the Euler targets (action[3:6] * 0.05 rad)."""
    import numpy as np
    from shortstop.calvin.reach import (
        CALVIN_ACTION_SCALE, CALVIN_ORIENTATION_SCALE, _rotation_from_euler, _rotation_vector, step_joint_config,
    )
    from shortstop.calvin.robot_geometry import _panda_transforms, gripper_tip_position
    from shortstop.calvin.scene_manifest import START_JOINT_ANGLES
    q = np.asarray(START_JOINT_ANGLES, dtype=float)
    euler = np.array([3.0, -0.1, 1.5])
    row = np.array([0.1, -0.05, 0.08, 0.04, -0.02, 0.06, 1.0])
    q_next, euler_next = step_joint_config(q, row, euler)
    move = gripper_tip_position(q_next) - gripper_tip_position(q)
    assert np.allclose(move, row[:3] * CALVIN_ACTION_SCALE, atol=2e-4)
    turn = _rotation_vector(_panda_transforms(q_next)[-1][:3, :3] @ _panda_transforms(q)[-1][:3, :3].T)
    expected = _rotation_vector(_rotation_from_euler(euler_next) @ _rotation_from_euler(euler).T)
    assert np.allclose(euler_next, euler + row[3:6] * CALVIN_ORIENTATION_SCALE)
    assert np.allclose(turn, expected, atol=2e-4)
