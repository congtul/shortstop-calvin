import numpy as np

from shortstop.calvin.reach import CALVIN_ACTION_SCALE, arm_robustness_to_go, nominal_joint_trajectory, propagate_arm_tube
from shortstop.calvin.shield import (
    ArmCBFShield, ArmConfThreshShield, ArmMPCFilterShield, ArmReachOnlyShield, ArmRepairShield, ArmSTLMonitorShield,
    ArmSTLShield,
)
from shortstop.env import Obstacle
from shortstop.calvin.robot_geometry import (
    FLANGE_FRAME_INDEX, JOINT_LIMITS, N_JOINTS, gripper_tip_position, panda_frames, within_joint_limits,
)

# A real, well-within-JOINT_LIMITS Franka "ready" pose -- NOT np.zeros:
# joint 4 (index 3)'s own real range is entirely negative ([-3.0718,
# -0.0698], see robot_geometry.JOINT_LIMITS), so q=0 is itself already
# physically invalid for that joint alone. Every test below that exercises
# a shield's actual select() (which enforces JOINT_LIMITS, see
# ArmReachOnlyShield._trajectory_within_joint_limits) needs a
# genuinely valid starting config, not q=0.
Q_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])


def _straight_chunk(dx, horizon=4):
    """`dx`: the REAL Cartesian meters this step should actually move.
    Divides by CALVIN_ACTION_SCALE to get the raw task-chunk value
    _step_joint_config expects, so every call site's `dx=0.05` means "5cm
    of real motion," not 5cm-times-50."""
    step = np.array([dx / CALVIN_ACTION_SCALE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    return np.tile(step, (horizon, 1))


def test_arm_reach_only_shield_rejects_a_chunk_that_hits_an_obstacle():
    q = Q_HOME
    unsafe = _straight_chunk(0.05)
    safe = _straight_chunk(-0.05)

    # obstacle placed where the "unsafe" chunk's gripper ends up
    from shortstop.calvin.reach import propagate_arm_tube
    from shortstop.calvin.robot_geometry import FLANGE_FRAME_INDEX
    tube = propagate_arm_tube(q, unsafe, w_bar=0.0, model_error=0.0)
    obstacle = Obstacle(center=tube[-1][FLANGE_FRAME_INDEX].center(), radius=0.05)

    shield = ArmReachOnlyShield(obstacles=[obstacle], w_bar=0.0, model_error=0.0)
    action, info = shield.select(q, [unsafe, safe], scores=[1.0, 0.0])

    assert info["admissible_mask"] == [False, True]
    assert np.allclose(action, safe)


def test_arm_stl_shield_rejects_within_margin_even_if_reach_only_would_accept():
    q = Q_HOME
    from shortstop.calvin.reach import propagate_arm_tube
    from shortstop.calvin.robot_geometry import FLANGE_FRAME_INDEX
    chunk = _straight_chunk(0.05)
    tube = propagate_arm_tube(q, chunk, w_bar=0.0, model_error=0.0)
    gripper_end = tube[-1][FLANGE_FRAME_INDEX].center()

    # obstacle just outside the true collision radius but inside STL's margin
    obstacle = Obstacle(center=gripper_end, radius=0.02)
    shield = ArmSTLShield(obstacles=[obstacle], w_bar=0.0, model_error=0.0, epsilon=0.05)
    _, info = shield.select(q, [chunk], scores=[1.0])
    assert info["admissible_mask"] == [False]


def test_arm_stl_monitor_shield_rejects_a_chunk_that_hits_an_obstacle():
    q = Q_HOME
    unsafe = _straight_chunk(0.05)
    safe = _straight_chunk(-0.05)

    tube = propagate_arm_tube(q, unsafe, w_bar=0.0, model_error=0.0)
    obstacle = Obstacle(center=tube[-1][FLANGE_FRAME_INDEX].center(), radius=0.05)

    shield = ArmSTLMonitorShield(obstacles=[obstacle], epsilon=0.0)
    action, info = shield.select(q, [unsafe, safe], scores=[1.0, 0.0])

    assert info["admissible_mask"] == [False, True]
    assert np.allclose(action, safe)


def test_arm_mpc_filter_shield_leaves_the_chunk_unchanged_when_no_obstacle_is_near():
    q = Q_HOME
    chunk = _straight_chunk(0.05)
    far_obstacle = Obstacle(center=np.array([100.0, 100.0, 100.0]), radius=0.02)

    shield = ArmMPCFilterShield(obstacles=[far_obstacle], w_bar=0.0, model_error=0.0)
    action, info = shield.select(q, [chunk], scores=[1.0])

    assert info["fallback"] is False
    assert info["intervened"] is False
    assert info["admissible_mask"] == [True]
    assert np.allclose(action[:, :3], chunk[:, :3], atol=1e-6)


def test_arm_mpc_filter_shield_corrects_a_chunk_that_hits_an_obstacle():
    """Verifies the corrected chunk is genuinely admissible under the TRUE
    (nonlinear) reachtube, not just the linearized model the QP itself
    solved against -- confirmed directly via
    propagate_arm_tube/arm_robustness_to_go on the CORRECTED chunk, the
    same ground-truth check every other shield's admissibility is judged
    by."""
    q = Q_HOME
    chunk = _straight_chunk(0.05)
    tube = propagate_arm_tube(q, chunk, w_bar=0.0, model_error=0.0)
    obstacle = Obstacle(center=tube[-1][FLANGE_FRAME_INDEX].center(), radius=0.05)

    shield = ArmMPCFilterShield(obstacles=[obstacle], w_bar=0.0, model_error=0.0)
    action, info = shield.select(q, [chunk], scores=[1.0])

    assert info["fallback"] is False
    assert info["intervened"] is True
    assert not np.allclose(action[:, :3], chunk[:, :3], atol=1e-6)

    corrected_tube = propagate_arm_tube(q, action, w_bar=0.0, model_error=0.0)
    assert arm_robustness_to_go(corrected_tube, [obstacle]) >= 0.0


def test_arm_mpc_filter_shield_enforces_joint_limits_even_with_no_obstacle():
    """A chunk that would drive a joint past JOINT_LIMITS over the horizon
    (no obstacle involved at all) must still get pulled back within range
    -- the arm's analogue of 2D's max_action_norm bound, checked directly
    via nominal_joint_trajectory/within_joint_limits on the TRUE (not
    linearized) resulting trajectory."""
    q = Q_HOME
    horizon = 10
    step = np.array([0.1 / CALVIN_ACTION_SCALE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    chunk = np.tile(step, (horizon, 1))
    assert not all(within_joint_limits(qk) for qk in nominal_joint_trajectory(q, chunk)[1:])

    shield = ArmMPCFilterShield(obstacles=[], w_bar=0.0, model_error=0.0)
    action, info = shield.select(q, [chunk], scores=[1.0])

    assert info["fallback"] is False
    assert info["intervened"] is True
    corrected_trajectory = nominal_joint_trajectory(q, action)
    assert all(within_joint_limits(qk) for qk in corrected_trajectory[1:])


def test_arm_mpc_filter_shield_falls_back_when_the_qp_is_infeasible():
    """An absurdly large obstacle radius (5m) centered on the chunk's own
    nominal endpoint can't be cleared by any correction within reach --
    the QP must report infeasible, not silently return a wrong/partial
    solution."""
    q = Q_HOME
    chunk = _straight_chunk(0.05)
    tube = propagate_arm_tube(q, chunk, w_bar=0.0, model_error=0.0)
    obstacle = Obstacle(center=tube[-1][FLANGE_FRAME_INDEX].center(), radius=5.0)

    shield = ArmMPCFilterShield(obstacles=[obstacle], w_bar=0.0, model_error=0.0)
    action, info = shield.select(q, [chunk], scores=[1.0])

    assert info["fallback"] is True
    assert info["admissible_mask"] == [False]
    assert np.allclose(action, np.zeros_like(chunk))


def test_arm_cbf_shield_leaves_the_chunk_unchanged_when_no_obstacle_is_near():
    q = Q_HOME
    chunk = _straight_chunk(0.05)
    far_obstacle = Obstacle(center=np.array([100.0, 100.0, 100.0]), radius=0.02)

    shield = ArmCBFShield(obstacles=[far_obstacle], w_bar=0.0, model_error=0.0)
    action, info = shield.select(q, [chunk], scores=[1.0])

    assert info["fallback"] is False
    assert info["intervened"] is False
    assert info["admissible_mask"] == [True]
    assert np.allclose(action[:, :3], chunk[:, :3], atol=1e-6)


def test_arm_cbf_shield_corrects_only_the_first_action_of_a_chunk_that_hits_an_obstacle():
    """Verifies CBF-Shield's defining property (see class docstring):
    unlike ArmMPCFilterShield (which corrects the WHOLE horizon in one
    QP), CBF is pointwise -- it may only ever touch candidates[0][0]. Rows
    1: must come back byte-for-byte identical to the nominal chunk, even
    though the SAME obstacle would make ArmMPCFilterShield correct every
    one of those rows too."""
    q = Q_HOME
    chunk = _straight_chunk(0.05, horizon=4)
    obstacle = _obstacle_one_step_from_violation(q, chunk)

    shield = ArmCBFShield(obstacles=[obstacle], w_bar=0.0, model_error=0.0, alpha=1.0)
    action, info = shield.select(q, [chunk], scores=[1.0])

    assert info["fallback"] is False
    assert info["intervened"] is True
    assert not np.allclose(action[0, :3], chunk[0, :3], atol=1e-6)
    assert np.allclose(action[1:, :3], chunk[1:, :3], atol=1e-9)  # untouched, unlike MPC-Filter

    # The corrected FIRST real step alone must be genuinely safe against
    # the true (nonlinear) reachtube -- the same ground-truth check every
    # other shield's own correction is judged by, restricted to horizon=1
    # since that's the only step this shield ever actually corrects.
    # Tolerance matches ArmMPCFilterShield's own resolve-test precedent
    # (real measured value there: ~-0.00068) -- a single linearization
    # pass against the true nonlinear chain always leaves some small,
    # expected residual, not exactly zero.
    corrected_tube = propagate_arm_tube(q, action[:1], w_bar=0.0, model_error=0.0)
    assert arm_robustness_to_go(corrected_tube, [obstacle]) >= -1e-3


def test_arm_cbf_shield_enforces_joint_limits_even_with_no_obstacle():
    """A single step that alone would drive a joint past JOINT_LIMITS (no
    obstacle involved at all) must still get pulled back within range --
    the arm's analogue of 2D's max_action_norm bound, checked directly
    via nominal_joint_trajectory/within_joint_limits on the TRUE (not
    linearized) resulting one-step trajectory. Only an extreme single-step
    delta (1.0 real meter -- far more than any realistic chunk row) is
    enough to trip a joint limit in just ONE step (a smaller, more modest
    delta stays within range even uncorrected, since JOINT_LIMITS' own
    range is wide); the QP's minimal-cost correction for this exact
    scenario happens to land essentially EXACTLY on the boundary (unlike
    ArmMPCFilterShield's own analogous test, which has 10 steps of slack
    to distribute the correction across), so the check below allows a
    tiny (1e-6 rad) floating-point/solver-tolerance margin rather than
    demanding an exact boolean pass."""
    q = Q_HOME
    step = np.array([1.0 / CALVIN_ACTION_SCALE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    chunk = np.tile(step, (2, 1))
    assert not all(within_joint_limits(qk) for qk in nominal_joint_trajectory(q, chunk[:1])[1:])

    shield = ArmCBFShield(obstacles=[], w_bar=0.0, model_error=0.0)
    action, info = shield.select(q, [chunk], scores=[1.0])

    assert info["fallback"] is False
    assert info["intervened"] is True
    corrected_trajectory = nominal_joint_trajectory(q, action[:1])
    q_next = corrected_trajectory[1]
    assert np.all(q_next >= JOINT_LIMITS[:, 0] - 1e-6)
    assert np.all(q_next <= JOINT_LIMITS[:, 1] + 1e-6)


def test_arm_cbf_shield_falls_back_when_the_qp_is_infeasible():
    """An absurdly large obstacle radius (5m) centered on the arm's own
    CURRENT flange position can't be cleared by any one-step correction
    -- the QP must report infeasible, not silently return a wrong/partial
    solution."""
    q = Q_HOME
    chunk = _straight_chunk(0.05)
    obstacle = Obstacle(center=panda_frames(q)[FLANGE_FRAME_INDEX], radius=5.0)

    shield = ArmCBFShield(obstacles=[obstacle], w_bar=0.0, model_error=0.0)
    action, info = shield.select(q, [chunk], scores=[1.0])

    assert info["fallback"] is True
    assert info["admissible_mask"] == [False]
    assert np.allclose(action, np.zeros_like(chunk))


def _obstacle_one_step_from_violation(q, chunk):
    """An obstacle the arm currently clears but would touch after the
    chunk's first step: on the flange's path, 0.15 m ahead (3 steps of 5 cm),
    so the flange (0.06 + 0.05 m margin) still clears it now."""
    tube = propagate_arm_tube(q, chunk, w_bar=0.0, model_error=0.0)
    obstacle = Obstacle(center=tube[3][FLANGE_FRAME_INDEX].center(), radius=0.05)
    standing_still = propagate_arm_tube(q, np.zeros((1, 7)), w_bar=0.0, model_error=0.0)
    assert arm_robustness_to_go(standing_still, [obstacle]) > 0.0
    assert arm_robustness_to_go(propagate_arm_tube(q, chunk[:1], w_bar=0.0, model_error=0.0), [obstacle]) < 0.0
    return obstacle


def test_arm_cbf_shield_smaller_alpha_keeps_more_margin():
    """From a currently safe state (h > 0) the CBF condition keeps
    h(next) >= (1 - alpha) h: alpha = 1 may go right up to the boundary,
    alpha = 0.3 must keep most of the current margin."""
    q = Q_HOME
    chunk = _straight_chunk(0.05, horizon=4)
    obstacle = _obstacle_one_step_from_violation(q, chunk)

    robustness = {}
    for alpha in (0.3, 1.0):
        shield = ArmCBFShield(obstacles=[obstacle], w_bar=0.0, model_error=0.0, alpha=alpha)
        action, _ = shield.select(q, [chunk], scores=[1.0])
        robustness[alpha] = arm_robustness_to_go(
            propagate_arm_tube(q, action[:1], w_bar=0.0, model_error=0.0), [obstacle])

    assert robustness[1.0] >= -1e-3  # boundary, modulo the one-step linearization
    assert robustness[0.3] > robustness[1.0] + 1e-3

def test_arm_stl_monitor_shield_picks_highest_score_among_admissible():
    q = Q_HOME
    a = _straight_chunk(0.05)
    b = _straight_chunk(0.03)
    # obstacle far from both candidates' paths -- both admissible
    obstacle = Obstacle(center=np.array([10.0, 10.0, 10.0]), radius=0.05)

    shield = ArmSTLMonitorShield(obstacles=[obstacle], epsilon=0.0)
    action, info = shield.select(q, [a, b], scores=[1.0, 5.0])

    assert info["admissible_mask"] == [True, True]
    assert np.allclose(action, b)  # higher score (5.0 > 1.0), not the first candidate


def test_arm_stl_monitor_shield_falls_back_when_every_candidate_hits_the_obstacle():
    q = Q_HOME
    a = _straight_chunk(0.05)
    b = _straight_chunk(0.05001)  # same direction, still ends up on the obstacle
    tube = propagate_arm_tube(q, a, w_bar=0.0, model_error=0.0)
    obstacle = Obstacle(center=tube[-1][FLANGE_FRAME_INDEX].center(), radius=0.05)

    shield = ArmSTLMonitorShield(obstacles=[obstacle], epsilon=0.0)
    action, info = shield.select(q, [a, b], scores=[1.0, 2.0])

    assert info["fallback"]
    assert np.allclose(action, np.zeros_like(a))


def test_arm_stl_monitor_shield_epsilon_is_a_real_tunable_margin_not_hardcoded_zero():
    """Verifies `epsilon` is a real, required constructor arg that
    actually gates admissibility (see ArmSTLMonitorShield's docstring on
    the paper's two conflicting readings of its threshold): a candidate
    whose nominal robustness is small and POSITIVE (0.04, computed
    directly via propagate_arm_tube/arm_robustness_to_go against this
    exact obstacle) must stay admissible at epsilon=0.0 ("rejects if
    negative", literal reading) but get rejected once epsilon is raised
    past it (0.05, "shared epsilon" reading) -- if epsilon were hardcoded
    to 0.0 internally instead, the second shield would wrongly keep it
    admissible too.
    """
    q = Q_HOME
    candidate = _straight_chunk(0.05)
    other = _straight_chunk(-0.05)  # moves away -- stays admissible at every epsilon tested here
    tube = propagate_arm_tube(q, candidate, w_bar=0.0, model_error=0.0)
    endpoint = tube[-1][FLANGE_FRAME_INDEX].center()
    # Offset chosen (verified via arm_find_counterexample, not hand-derived)
    # so arm_robustness_to_go(candidate's tube, [obstacle]) is a small
    # POSITIVE margin (~0.028): admissible at epsilon=0.0, rejected once
    # epsilon is raised past it. The binding primitive here is the
    # "fingertip" capsule (see propagate_arm_tube's docstring) -- NOT the
    # flange's own point-capsule, which uses LINK_RADIUS[-1]=0.06 rather
    # than FRAME_RADIUS[FLANGE_FRAME_INDEX]=0.20, so it's not the tightest
    # constraint at this offset.
    obstacle = Obstacle(center=endpoint + np.array([0.14, 0.0, 0.0]), radius=0.05)

    lenient = ArmSTLMonitorShield(obstacles=[obstacle], epsilon=0.0)
    action, info = lenient.select(q, [candidate, other], scores=[2.0, 1.0])
    assert info["admissible_mask"] == [True, True]
    assert np.allclose(action, candidate)  # higher score, still admissible at epsilon=0.0

    strict = ArmSTLMonitorShield(obstacles=[obstacle], epsilon=0.05)
    action, info = strict.select(q, [candidate, other], scores=[2.0, 1.0])
    assert info["admissible_mask"] == [False, True]  # same candidate now rejected
    assert np.allclose(action, other)


def test_no_shield_defines_recertify_or_resolve_anymore():
    """The harness makes one decision per Propose call, for every baseline
    uniformly -- there is no mid-window recertify()/resolve() mechanism
    (see shortstop.calvin.experiment). This pins that no shield class in this
    module defines either method (a stray leftover would silently
    reintroduce mid-window dispatch, since run_calvin_shielded_subtask's
    own `hasattr` checks would pick it back up)."""
    shields = [
        ArmReachOnlyShield(obstacles=[], w_bar=0.0, model_error=0.0, horizon_steps=8),
        ArmSTLShield(obstacles=[], w_bar=0.0, model_error=0.0, horizon_steps=8),
        ArmSTLMonitorShield(obstacles=[], epsilon=0.0, horizon_steps=8),
        ArmRepairShield(obstacles=[], w_bar=0.0, model_error=0.0, horizon_steps=8),
        ArmMPCFilterShield(obstacles=[], w_bar=0.0, model_error=0.0, horizon_steps=8),
        ArmCBFShield(obstacles=[], w_bar=0.0, model_error=0.0),
        ArmConfThreshShield(disagreement_threshold=1.0, horizon_steps=8),
    ]
    for shield in shields:
        assert not hasattr(shield, "recertify"), f"{type(shield).__name__} still defines recertify()"
        assert not hasattr(shield, "resolve"), f"{type(shield).__name__} still defines resolve()"


def test_arm_repair_shield_fixes_a_rejected_candidate_and_still_certifies_it():
    q = Q_HOME
    from shortstop.calvin.reach import propagate_arm_tube
    from shortstop.calvin.robot_geometry import FLANGE_FRAME_INDEX
    chunk = _straight_chunk(0.05)
    tube = propagate_arm_tube(q, chunk, w_bar=0.0, model_error=0.0)
    obstacle = Obstacle(center=tube[-1][FLANGE_FRAME_INDEX].center(), radius=0.05)

    # trust_region=0.3 gives Q_HOME's own Jacobian geometry enough room for
    # the 3-iteration repair to actually converge; verified directly (not
    # guessed) before picking this value.
    shield = ArmRepairShield(
        obstacles=[obstacle], w_bar=0.0, model_error=0.0, epsilon=0.02,
        trust_region=0.3, step_size=0.1, max_repair_iters=3,
    )
    action, info = shield.select(q, [chunk], scores=[1.0])

    assert info["repair_attempted"]
    assert info["repair_succeeded"]
    assert info["admissible_mask"] == [True]
    assert not np.allclose(action, chunk)  # actually got modified


def test_repair_direction_matches_the_naive_raw_direction_when_witness_is_the_tcp():
    """_repair_direction maps the Cartesian push-away vector through
    `M = witness_jacobian @ pinv(J_tcp)` (actions command the TCP, see
    reach._step_joint_config). When the witness IS the TCP -- the
    fingertip capsule's far end -- M ~= identity, so the direction must
    reduce to the raw normalize(witness - obstacle.center)."""
    q = Q_HOME
    chunk = _straight_chunk(0.05)
    tip = gripper_tip_position(q)
    axis = tip - panda_frames(q)[FLANGE_FRAME_INDEX]
    # Obstacle beyond the tip along the flange axis, slightly off it: the
    # fingertip capsule's closest point is its tip end (t = 1).
    obstacle_center = tip + 0.3 * axis + np.array([0.01, 0.02, 0.0])
    obstacle = Obstacle(center=obstacle_center, radius=0.05)
    shield = ArmRepairShield(obstacles=[obstacle], w_bar=0.0, model_error=0.0, epsilon=0.02)
    ce = {"step": 0, "frame": "fingertip", "obstacle": obstacle, "witness": tip}

    raw_direction = ce["witness"] - ce["obstacle"].center
    raw_direction = raw_direction / np.linalg.norm(raw_direction)
    projected_direction = shield._repair_direction(q, chunk, ce)

    assert np.allclose(projected_direction, raw_direction, atol=1e-3)


def test_repair_direction_differs_from_the_naive_raw_direction_for_a_non_flange_witness():
    """For a witness that ISN'T the flange, the Jacobian projection
    generally does NOT reduce to the raw Cartesian push-away vector --
    this pins that the projection actually changes behavior for that
    case, not just leaving it a silent no-op.
    Obstacle centered NEAR (not exactly at, see the flange test's own
    comment on why exact placement is degenerate for this formula) frame
    index 4 (a mid-chain joint, not the flange) -- hand-built
    counterexample dict, same reasoning as the flange test above."""
    q = Q_HOME
    chunk = _straight_chunk(0.05)
    non_flange_frame_index = 4
    witness_point = panda_frames(q)[non_flange_frame_index]
    obstacle_center = witness_point + np.array([0.05, 0.03, -0.02])
    obstacle = Obstacle(center=obstacle_center, radius=0.05)
    shield = ArmRepairShield(obstacles=[obstacle], w_bar=0.0, model_error=0.0, epsilon=0.02)
    ce = {"step": 1, "frame": non_flange_frame_index, "obstacle": obstacle, "witness": witness_point}

    raw_direction = ce["witness"] - ce["obstacle"].center
    raw_direction = raw_direction / np.linalg.norm(raw_direction)
    projected_direction = shield._repair_direction(q, chunk, ce)

    assert not np.allclose(projected_direction, raw_direction, atol=1e-2)


def test_repair_direction_falls_back_to_the_raw_direction_when_the_jacobian_projection_degenerates(monkeypatch):
    """Forces the projected_norm < 1e-9 branch directly (monkeypatching
    _witness_jacobian to return an all-zero (3,7) matrix -- a stand-in for
    a witness primitive whose motion is, to first order, fully decoupled
    from the flange's own action at this configuration) instead of
    hunting for a real physical near-singularity. Must return the raw
    normalize(witness - obstacle.center) direction for this one call --
    not raise, not divide-by-zero into nan."""
    q = Q_HOME
    chunk = _straight_chunk(0.05)
    witness_point = panda_frames(q)[4] + np.array([0.1, 0.0, 0.0])
    obstacle_center = panda_frames(q)[4]
    obstacle = Obstacle(center=obstacle_center, radius=0.05)
    shield = ArmRepairShield(obstacles=[obstacle], w_bar=0.0, model_error=0.0, epsilon=0.02)
    ce = {"step": 1, "frame": 4, "obstacle": obstacle, "witness": witness_point}

    monkeypatch.setattr(shield, "_witness_jacobian", lambda *a, **k: np.zeros((3, 7)))
    direction = shield._repair_direction(q, chunk, ce)

    expected = witness_point - obstacle_center
    expected = expected / np.linalg.norm(expected)
    assert np.allclose(direction, expected)
    assert np.isfinite(direction).all()


def test_arm_repair_shield_falls_back_when_repair_cannot_fix_it_in_time():
    q = Q_HOME
    chunk = _straight_chunk(0.05)
    from shortstop.calvin.reach import propagate_arm_tube
    from shortstop.calvin.robot_geometry import FLANGE_FRAME_INDEX
    tube = propagate_arm_tube(q, chunk, w_bar=0.0, model_error=0.0)
    obstacle = Obstacle(center=tube[-1][FLANGE_FRAME_INDEX].center(), radius=0.05)

    # tiny trust region + tiny step -> repair can't move far enough to clear
    shield = ArmRepairShield(
        obstacles=[obstacle], w_bar=0.0, model_error=0.0, epsilon=0.02,
        trust_region=1e-6, step_size=1e-6, max_repair_iters=1,
    )
    action, info = shield.select(q, [chunk], scores=[1.0])
    assert info["fallback"]
    assert np.allclose(action, np.zeros_like(chunk))




def _endpoint(q, chunk):
    tube = propagate_arm_tube(q, chunk, w_bar=0.0, model_error=0.0)
    return tube[-1][FLANGE_FRAME_INDEX].center()


def test_arm_conf_thresh_shield_rejects_the_outlier_but_keeps_the_agreeing_candidates():
    q = np.zeros(N_JOINTS)
    agree_a = _straight_chunk(0.05)
    agree_b = _straight_chunk(0.05001)  # same direction, tiny float variation -> near-identical endpoint
    outlier = _straight_chunk(-0.05)  # opposite direction -> far from the other two

    distances_from_centroid = [
        np.linalg.norm(e - np.mean([_endpoint(q, c) for c in (agree_a, agree_b, outlier)], axis=0))
        for e in [_endpoint(q, c) for c in (agree_a, agree_b, outlier)]
    ]
    threshold = (max(distances_from_centroid[:2]) + distances_from_centroid[2]) / 2

    shield = ArmConfThreshShield(disagreement_threshold=threshold, horizon_steps=4)
    action, info = shield.select(q, [agree_a, agree_b, outlier], scores=[1.0, 2.0, 3.0])

    assert info["admissible_mask"] == [True, True, False]
    assert np.allclose(action, agree_b)  # admissible with the higher score (2.0 > 1.0)


def test_arm_conf_thresh_shield_picks_highest_score_among_admissible():
    q = np.zeros(N_JOINTS)
    a = _straight_chunk(0.05)
    b = _straight_chunk(0.05001)  # close enough to agree at a generous threshold

    shield = ArmConfThreshShield(disagreement_threshold=10.0, horizon_steps=4)
    action, info = shield.select(q, [a, b], scores=[5.0, 1.0])

    assert info["admissible_mask"] == [True, True]
    assert np.allclose(action, a)


def test_arm_conf_thresh_shield_falls_back_when_every_candidate_disagrees():
    q = np.zeros(N_JOINTS)
    a = _straight_chunk(0.05)
    b = _straight_chunk(-0.05)

    shield = ArmConfThreshShield(disagreement_threshold=1e-9, horizon_steps=4)
    action, info = shield.select(q, [a, b], scores=[1.0, 2.0])

    assert info["fallback"]
    assert np.allclose(action, np.zeros_like(a))


def test_arm_conf_thresh_shield_measures_disagreement_only_over_the_first_horizon_steps_rows():
    """`horizon_steps` here is the Conf-Thresh rejection criterion's own
    parameter, not the harness's execution cadence: `a` and `b` are
    IDENTICAL for their first 2 rows, then diverge sharply on rows 3-4 --
    rows that
    would never be certified if horizon_steps=2. With horizon_steps=2 both
    candidates' endpoints coincide exactly (disagreement=0, trivially
    admissible at any threshold); with horizon_steps=4 (the full chunk)
    they end up ~1.1m apart (disagreement>0, rejected at a tight
    threshold). Same two candidates, same threshold -- only horizon_steps
    changes, so a different verdict proves the endpoint really is being
    computed over the truncated prefix, not silently still the full chunk.
    """
    q = np.zeros(N_JOINTS)
    # Raw values divided by CALVIN_ACTION_SCALE (see _straight_chunk's own
    # docstring above) so the REAL motion -- and thus this test's own
    # disagreement/threshold reasoning below -- matches actual meters, not
    # scaled units.
    a = np.array([[0.05 / CALVIN_ACTION_SCALE, 0, 0, 0, 0, 0, 0]] * 4)
    b = np.array([[0.05 / CALVIN_ACTION_SCALE, 0, 0, 0, 0, 0, 0], [0.05 / CALVIN_ACTION_SCALE, 0, 0, 0, 0, 0, 0],
                  [-0.5 / CALVIN_ACTION_SCALE, 0, 0, 0, 0, 0, 0], [-0.5 / CALVIN_ACTION_SCALE, 0, 0, 0, 0, 0, 0]])
    assert np.allclose(a[:2], b[:2])  # identical prefix by construction
    assert not np.allclose(_endpoint(q, a), _endpoint(q, b))  # but diverge by the full chunk's end

    threshold = 0.3  # strictly between 0 (prefix disagreement) and ~0.55 (full-chunk disagreement)

    shield_prefix = ArmConfThreshShield(disagreement_threshold=threshold, horizon_steps=2)
    shield_full = ArmConfThreshShield(disagreement_threshold=threshold, horizon_steps=4)

    _, info_prefix = shield_prefix.select(q, [a, b], scores=[1.0, 1.0])
    _, info_full = shield_full.select(q, [a, b], scores=[1.0, 1.0])

    assert info_prefix["admissible_mask"] == [True, True]
    assert info_full["admissible_mask"] == [False, False]


def test_stl_monitor_admits_everything_with_no_obstacles():
    """Regression: arm_step_robustness took min() over an empty obstacle
    list and crashed when a shield ran on an obstacle-free scene."""
    shield = ArmSTLMonitorShield(obstacles=[], epsilon=0.0, horizon_steps=4)
    candidates = [np.zeros((4, 7)), np.ones((4, 7)) * 0.01]
    chunk, info = shield.select(Q_HOME, candidates, scores=[0.0, 1.0])
    assert info["admissible_mask"] == [True, True]
    assert np.array_equal(chunk, candidates[1])


def test_arm_cbf_shield_with_steps_filters_each_executed_row_along_the_prediction():
    """steps=4: rows 0-3 are each filtered at the predicted state after the
    rows before them, so the 4-step prefix the harness executes stays clear;
    rows from `steps` on are left as proposed."""
    q = Q_HOME
    chunk = _straight_chunk(0.05, horizon=6)
    obstacle = _obstacle_one_step_from_violation(q, chunk)

    one = ArmCBFShield(obstacles=[obstacle], w_bar=0.0, model_error=0.0, alpha=1.0, steps=1)
    four = ArmCBFShield(obstacles=[obstacle], w_bar=0.0, model_error=0.0, alpha=1.0, steps=4)
    action_one, _ = one.select(q, [chunk], scores=[1.0])
    action_four, info = four.select(q, [chunk], scores=[1.0])

    assert info["status"] == "corrected"
    assert np.allclose(action_four[4:, :3], chunk[4:, :3])
    prefix = lambda action: arm_robustness_to_go(  # noqa: E731
        propagate_arm_tube(q, action[:4], w_bar=0.0, model_error=0.0), [obstacle])
    assert prefix(action_one) < 0.0  # only row 0 filtered: rows 1-3 run into the obstacle
    assert prefix(action_four) >= -1e-3


def _blocked_chunk_and_obstacle(horizon=6, offset=0.03):
    """A chunk whose end hits an obstacle `offset` m to its side (its first
    rows are clear)."""
    chunk = _straight_chunk(0.03, horizon)
    tube = propagate_arm_tube(Q_HOME, chunk, w_bar=0.0, model_error=0.0)
    center = tube[-1][FLANGE_FRAME_INDEX].center() + np.array([0.0, offset, 0.0])
    return chunk, Obstacle(center=center, radius=0.03 if offset else 0.04)


def test_committed_rows_are_never_modified_by_any_correcting_shield():
    """delay_aware: rows before `fixed_prefix` already run on the arm; repair,
    MPC and CBF may only change the rows after them."""
    chunk, obstacle = _blocked_chunk_and_obstacle()
    shields = [
        ArmRepairShield(obstacles=[obstacle], w_bar=0.0, model_error=0.0, epsilon=0.003, repair_mode="polyak",
                        repair_margin=0.001, max_linearized_iters=5, horizon_steps=6),
        ArmMPCFilterShield(obstacles=[obstacle], w_bar=0.0, model_error=0.0, horizon_steps=6),
        ArmCBFShield(obstacles=[obstacle], w_bar=0.0, model_error=0.0, alpha=0.5, steps=6),
    ]
    for shield in shields:
        shield.fixed_prefix = 2
        action, info = shield.select(Q_HOME, [chunk.copy()], scores=[1.0])
        assert not info["fallback"], type(shield).__name__
        assert np.allclose(action[:2], chunk[:2]), type(shield).__name__
        assert not np.allclose(action[2:6, :3], chunk[2:6, :3]), type(shield).__name__


def test_repair_fails_when_the_violation_lies_in_the_committed_rows():
    chunk, obstacle = _blocked_chunk_and_obstacle(horizon=4, offset=0.0)
    shield = ArmRepairShield(obstacles=[obstacle], w_bar=0.0, model_error=0.0, epsilon=0.003, repair_mode="polyak",
                             repair_margin=0.001, max_linearized_iters=5, horizon_steps=4)
    shield.fixed_prefix = 4
    _, info = shield.select(Q_HOME, [chunk.copy()], scores=[1.0])
    assert info["fallback"] and not info["repair_succeeded"]


def test_reach_only_ablation_certifies_exactly_like_shortstop():
    """The ablation ladder's Reach-only rung (make_shield "reach_only") accepts
    the same candidates ShortStop certifies as proposed; only repair differs."""
    from shortstop.calvin.baseline_runner import make_shield
    from shortstop.calvin.config import SHORTSTOP_PARAMS

    rng = np.random.default_rng(0)
    reach, shortstop = make_shield("reach_only", {}), make_shield("shortstop", SHORTSTOP_PARAMS)
    _, obstacle = _blocked_chunk_and_obstacle(horizon=4, offset=0.03)
    reach.obstacles = shortstop.obstacles = [obstacle]
    candidates = [_straight_chunk(0.03 * rng.uniform(-1.0, 1.0)) for _ in range(40)]
    reach_mask = [reach._admissible(Q_HOME, c) for c in candidates]
    assert reach_mask == [shortstop._certify(Q_HOME, c)[1] for c in candidates]
    assert any(reach_mask) and not all(reach_mask)


def test_without_g_a_the_first_admissible_candidate_in_policy_order_is_chosen():
    """use_progress=False scores every candidate 0: Select keeps the policy's order."""
    unsafe, obstacle = _blocked_chunk_and_obstacle(horizon=4, offset=0.0)
    safe_a, safe_b = _straight_chunk(-0.02), _straight_chunk(-0.04)
    shield = ArmReachOnlyShield(obstacles=[obstacle], w_bar=0.0, model_error=0.0)
    action, info = shield.select(Q_HOME, [unsafe, safe_a, safe_b], scores=[0.0, 0.0, 0.0])
    assert info["admissible_mask"] == [False, True, True]
    assert np.allclose(action, safe_a)
