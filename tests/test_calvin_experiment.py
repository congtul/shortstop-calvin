"""Exercises shortstop.calvin.experiment's control-flow logic (replan
cycling, ground-truth violation check, stop-on-violation, task_oracle
success check) against fakes standing in for HulcWrapper/task_oracle/
lang_embeddings -- no real CALVIN/mdt_policy session involved. See
docs/CALVIN_SETUP.md for the real interface this mirrors.
"""
import copy
import functools

import numpy as np
import pytest
import torch

from shortstop.calvin.shield import ArmConfThreshShield, ArmSTLMonitorShield
from shortstop.calvin.reach import CALVIN_ACTION_SCALE, propagate_arm_tube
from shortstop.calvin.experiment import (
    _base_transform_from_env,
    _clearance,
    run_calvin_shielded_subtask,
    run_calvin_unshielded_sequence,
    run_calvin_unshielded_subtask,
    restore_env_snapshot,
    snapshot_env,
)
from shortstop.calvin.metrics import build_fixed_cohort_slots, fixed_cohort_rates
from shortstop.env import Obstacle
from shortstop.calvin.robot_geometry import FLANGE_FRAME_INDEX, N_JOINTS, panda_frames

SUBTASK = "fake_subtask"

# A real, well-within-JOINT_LIMITS Franka "ready" pose -- NOT np.zeros: see
# tests/test_arm_shield.py's own Q_HOME for why q=0 is itself physically
# invalid for joint 4 alone, which matters for any test exercising a real
# shield's select() (now enforcing JOINT_LIMITS).
Q_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])


class _FakeSnapshotTarget:
    """Stand-in for calvin_env's own Robot/Scene serialize()/reset_from_
    storage() pair -- captures/restores a fixed list of attributes on
    `outer` (the enclosing FakeEnv), deep-copied so a later mutation of
    `outer`'s own live attribute can't corrupt an already-taken snapshot
    (matches PyBullet's own real serialize(), which reads a fresh
    getJointStates()/getBasePositionAndOrientation() snapshot every call,
    not a live reference)."""

    def __init__(self, outer, fields, **extra_attrs):
        self.outer = outer
        self.fields = fields
        for name, value in extra_attrs.items():
            setattr(self, name, value)

    def serialize(self):
        return {field: copy.deepcopy(getattr(self.outer, field)) for field in self.fields}

    def reset_from_storage(self, data):
        for field, value in data.items():
            setattr(self.outer, field, copy.deepcopy(value))


class _FakeBulletClient:
    def stepSimulation(self, physicsClientId=None):
        del physicsClientId


class _FakePrivilegedEnv:
    """Minimal stand-in for env.env (calvin_env's own PlayTableSimEnv) --
    only the serialize()/robot/scene/p/cid surface shortstop.
    calvin.experiment.restore_env_snapshot and _privileged_rollout_violates
    actually touch, round-tripping the OUTER FakeEnv's own mutable state
    (joint_angles/step_count/gripper_action/last_gripper_action/
    scene_positions) the same way the real PlayTableSimEnv.serialize()/
    reset_from_storage() round-trips real robot+scene state."""

    def __init__(self, outer):
        # base_position/base_orientation: identity, matching every other
        # FakeEnv-based test's own assumption that g(a)'s base-frame
        # correction is a no-op here (see
        # test_base_transform_from_env_defaults_to_identity_without_env_
        # dot_robot for the case that's genuinely missing this chain).
        self.robot = _FakeSnapshotTarget(
            outer, ["joint_angles", "step_count", "gripper_action", "last_gripper_action"],
            base_position=(0.0, 0.0, 0.0), base_orientation=(0.0, 0.0, 0.0, 1.0),
        )
        self.scene = _FakeSnapshotTarget(outer, ["scene_positions"])
        self.p = _FakeBulletClient()
        self.cid = 0

    def serialize(self):
        return {"robot": self.robot.serialize(), "scene": self.scene.serialize()}


class FakeEnv:
    """joint_angles[0] += action[0] each step -- just enough dynamics to
    move the sphere-chain in a controlled, predictable way. `scene_positions`
    (name -> [x,y,z]) and `gripper_action` are read by get_info()/get_obs()
    for the shielded-harness tests (calvin_progress_scores's target-object
    lookup, and gripper-action handling) -- unused/benign for the plain
    unshielded tests below. `self.env` is a fake stand-in for env.env --
    see _FakePrivilegedEnv -- needed by
    shortstop.calvin.experiment._privileged_rollout_violates's own
    fork-and-restore ground-truth check for intervention_precision."""

    def __init__(self):
        self.joint_angles = np.zeros(N_JOINTS)
        self.step_count = 0
        self.gripper_action = 1.0
        self.gripper_width = 0.0  # closed by default
        self.scene_positions = {}
        self.last_gripper_action = None
        self.env = _FakePrivilegedEnv(self)

    def reset(self, robot_obs=None, scene_obs=None):
        self.joint_angles = np.zeros(N_JOINTS)
        self.step_count = 0
        return self.get_obs()

    def get_obs(self):
        robot_obs_raw = np.zeros(15, dtype=np.float32)
        robot_obs_raw[6] = self.gripper_width
        robot_obs_raw[7:14] = self.joint_angles
        robot_obs_raw[14] = self.gripper_action
        return {"robot_obs_raw": torch.tensor(robot_obs_raw)}

    def get_info(self):
        return {
            "step_count": self.step_count,
            "scene_info": {
                "movable_objects": {
                    name: {"current_pos": pos} for name, pos in self.scene_positions.items()
                },
            },
        }

    def step(self, action_tensor):
        action = action_tensor.detach().cpu().numpy()
        self.last_gripper_action = float(action[6])
        self.joint_angles = self.joint_angles.copy()
        self.joint_angles[0] += action[0]
        self.step_count += 1
        return self.get_obs(), 0.0, False, self.get_info()


class FakeTaskOracle:
    def __init__(self, success_after_steps):
        self.success_after_steps = success_after_steps

    def get_task_info_for_set(self, start_info, current_info, subtask_set):
        if current_info["step_count"] >= self.success_after_steps:
            return set(subtask_set)
        return set()


class FakeTaskOracleWithTasks(FakeTaskOracle):
    """Adds `.tasks` ({subtask_name: functools.partial(fn, obj_name, ...)},
    matching calvin_env.envs.tasks.Tasks's own shape) -- needed by
    shortstop.calvin.progress.calvin_progress_scores for the shielded
    harness, not by the plain unshielded one."""

    def __init__(self, success_after_steps, tasks):
        super().__init__(success_after_steps)
        self.tasks = tasks


class FakePolicy:
    """Always proposes 1 candidate chunk moving joint 0 by `delta`/step."""

    def __init__(self, delta=0.1, horizon=10):
        self.delta = delta
        self.horizon = horizon

    def propose(self, observation):
        del observation
        chunk = np.zeros((self.horizon, 7))
        chunk[:, 0] = self.delta
        return [chunk]


class FakeMultiCandidatePolicy:
    """Always proposes the same fixed list of candidate chunks, regardless
    of observation -- deterministic stand-in for a K-candidate policy."""

    def __init__(self, chunks):
        self.chunks = chunks

    def propose(self, observation):
        del observation
        return list(self.chunks)


def _joint0_chunk(delta, horizon=10):
    chunk = np.zeros((horizon, 7))
    chunk[:, 0] = delta
    return chunk


def _predicted_endpoint(joint_angles, chunk):
    tube = propagate_arm_tube(joint_angles, chunk, w_bar=0.0, model_error=0.0)
    return tube[-1]["fingertip"].b  # the TCP, the point an action commands


class FakeLangEmbeddings:
    def get_lang_goal(self, task):
        return {"lang": task}


VAL_ANNOTATIONS = {SUBTASK: [SUBTASK]}


def test_unshielded_subtask_succeeds_when_never_flagged():
    env = FakeEnv()
    env.reset()
    result = run_calvin_unshielded_subtask(
        env, FakePolicy(), FakeTaskOracle(success_after_steps=12), FakeLangEmbeddings(), SUBTASK, VAL_ANNOTATIONS,
        ep_len=360, replan_steps=10, obstacles=None,
    )
    assert result["violated"] is False
    assert result["reached"] is True
    assert result["min_clearance"] is None
    assert result["steps_taken"] == 12
    assert len(result["propose_latencies_ms"]) == 2  # 2 replan cycles of up to 10 rows each to cover 12 steps
    assert all(t >= 0.0 for t in result["propose_latencies_ms"])


def test_unshielded_subtask_stops_at_violation_and_never_reaches():
    env = FakeEnv()
    env.reset()
    # obstacle placed exactly where the arm is after 6 steps -- well
    # before the task_oracle would succeed at step 12
    obstacle_joint_angles = np.zeros(N_JOINTS)
    obstacle_joint_angles[0] = 0.1 * 6
    obstacle = Obstacle(center=panda_frames(obstacle_joint_angles)[-1], radius=0.02)

    result = run_calvin_unshielded_subtask(
        env, FakePolicy(), FakeTaskOracleWithTasks(success_after_steps=12, tasks={}), FakeLangEmbeddings(), SUBTASK,
        VAL_ANNOTATIONS, ep_len=360, replan_steps=10,
        obstacles=obstacle,
    )
    assert result["violated"] is True
    assert result["reached"] is False  # violated => never gets a chance to also succeed
    assert result["min_clearance"] <= 0.0  # violated => clearance hit zero/negative


def test_base_transform_from_env_defaults_to_identity_without_env_dot_robot():
    """A bare object with no `.env.robot` chain at all -- _base_transform_
    from_env must fall back to identity rather than raising. (FakeEnv
    itself DOES have a `.env.robot` chain, for the privileged-rollout
    snapshot mechanism -- but its own base_position/base_orientation are
    set to this SAME identity, so every other FakeEnv-based test's
    assumption that g(a)'s base-frame correction is a no-op still holds,
    just via a populated-but-identity chain instead of a missing one.)"""
    class _BareEnv:
        pass

    base_position, base_orientation = _base_transform_from_env(_BareEnv())
    assert base_position == (0.0, 0.0, 0.0)
    assert base_orientation == (0.0, 0.0, 0.0, 1.0)


def test_base_transform_from_env_reads_the_real_robot_base_pose_when_present():
    class _FakeRobot:
        base_position = [-0.34, -0.46, 0.24]
        base_orientation = [0.0, 0.0, 0.0, 1.0]

    class _FakeInnerEnv:
        robot = _FakeRobot()

    class _FakeWrapper:
        env = _FakeInnerEnv()

    base_position, base_orientation = _base_transform_from_env(_FakeWrapper())
    assert base_position == [-0.34, -0.46, 0.24]
    assert base_orientation == [0.0, 0.0, 0.0, 1.0]


def test_clearance_tracks_the_real_gripper_width_not_a_fixed_worst_case_bound():
    """Verifies _clearance() computes clearance from the actual gripper
    width at each step, not a fixed worst-case (fully open) assumption --
    a fully-open assumption would be conservative but wrong when the
    gripper is actually closed or partway: an obstacle placed exactly
    where an OPEN finger's own tip would be must read as clear when the
    gripper is actually closed, and as violated once it's actually open
    (same joint_angles both times -- only gripper_width differs)."""
    from shortstop.calvin.robot_geometry import finger_tip_capsules

    env = FakeEnv()
    env.reset()
    q = np.array([0.0, 0.5, 0.0, -1.0, 0.0, 1.0, 0.0])  # unfolded -- avoids q=0's own degenerate self-overlap
    env.joint_angles = q.copy()
    (_, left_far), _ = finger_tip_capsules(q, gripper_width=0.08)
    obstacle = Obstacle(center=left_far, radius=0.01)

    env.gripper_width = 0.0
    assert _clearance(env.get_obs(), obstacle) > 0.0  # closed -- finger nowhere near the obstacle

    env.gripper_width = 0.08
    assert _clearance(env.get_obs(), obstacle) <= 0.0  # open -- finger tip is exactly at the obstacle


def test_violation_rate_and_success_rate_differ_with_vs_without_obstacle():
    """The exact comparison requested: same policy/task, run once with no
    obstacle at all and once with a privileged obstacle placed on the
    arm's own path -- confirm success_rate drops and violation_rate rises.
    """
    def make_sequence_result(obstacle):
        env = FakeEnv()
        return run_calvin_unshielded_sequence(
            env, FakePolicy(), FakeTaskOracleWithTasks(success_after_steps=12, tasks={}), FakeLangEmbeddings(),
            initial_condition={}, eval_sequence=[SUBTASK], val_annotations=VAL_ANNOTATIONS,
            get_env_state_for_initial_condition=lambda ic: (None, None),
            ep_len=360, replan_steps=10,
            obstacles=obstacle,
        )

    obstacle_joint_angles = np.zeros(N_JOINTS)
    obstacle_joint_angles[0] = 0.1 * 6
    obstacle = Obstacle(center=panda_frames(obstacle_joint_angles)[-1], radius=0.02)

    n_sequences = 20
    results_without = [make_sequence_result(None) for _ in range(n_sequences)]
    results_with = [make_sequence_result(obstacle) for _ in range(n_sequences)]

    slots_without = build_fixed_cohort_slots(results_without, subtasks_per_sequence=1)
    slots_with = build_fixed_cohort_slots(results_with, subtasks_per_sequence=1)

    violation_without, success_without = fixed_cohort_rates(slots_without)
    violation_with, success_with = fixed_cohort_rates(slots_with)

    assert violation_without == 0.0
    assert success_without == 1.0
    assert violation_with == 1.0  # every run takes the identical deterministic path through the obstacle
    assert success_with == 0.0


def test_shielded_subtask_executes_the_shields_higher_scoring_admissible_candidate():
    env = FakeEnv()
    env.reset()

    positive_delta = _joint0_chunk(0.1)
    negative_delta = _joint0_chunk(-0.1)
    policy = FakeMultiCandidatePolicy([positive_delta, negative_delta])

    # target object sits exactly at the +delta candidate's own predicted
    # endpoint -> its g(a) (-distance) is unambiguously the higher of the two
    target_pos = _predicted_endpoint(np.zeros(N_JOINTS), positive_delta)
    env.scene_positions = {"block_red": target_pos.tolist()}
    task_oracle = FakeTaskOracleWithTasks(
        success_after_steps=10**9,  # never triggers within this single-decision window
        tasks={SUBTASK: functools.partial(lambda *a, **k: None, "block_red")},
    )

    shield = ArmConfThreshShield(disagreement_threshold=10.0, horizon_steps=10)  # generous -> both candidates admissible

    result = run_calvin_shielded_subtask(
        env, policy, task_oracle, FakeLangEmbeddings(), SUBTASK, VAL_ANNOTATIONS, shield,
        ep_len=10, replan_steps=10, horizon_steps=10, obstacles=None,
    )

    assert result["n_decisions"] == 1
    assert result["n_activated"] == 0  # nothing rejected -- both admissible
    assert env.joint_angles[0] > 0  # executed the +delta candidate (higher g(a), closer to target)


def test_shielded_subtask_wires_the_obstacle_into_an_obstacle_aware_shield_before_its_first_decision():
    """An obstacle-aware shield (has a `.obstacles` attribute, e.g.
    ArmSTLMonitorShield) must have it set to the ACTUALLY placed obstacle
    before its very first select() call of a subtask -- not left over
    from whatever it was constructed with. Checked by constructing the
    shield with a far-away placeholder obstacle, then placing the real
    obstacle exactly on one candidate's own predicted endpoint: that candidate
    must be rejected on the very first decision; if the shield were still
    certifying against the stale far-away placeholder, it would wrongly
    pass as admissible.

    Starts from Q_HOME (not np.zeros -- see this module's own Q_HOME
    docstring) and moves along y (not x) for `other`: select() also
    enforces JOINT_LIMITS (see ArmReachOnlyShield._trajectory_within_
    joint_limits), and a purely-x displacement large enough to clear the
    flange's own inflation radius drives some joint well past its real
    limit over 10 repeated rows -- verified directly (not guessed) that
    this dx/dy pair stays within JOINT_LIMITS for both candidates while
    still separating geometrically."""
    env = FakeEnv()
    env.reset()
    env.joint_angles = Q_HOME.copy()

    def _xyz_chunk(dx, dy, dz, horizon=10):
        """dx/dy/dz: REAL Cartesian meters per step. Divided by
        CALVIN_ACTION_SCALE to get the raw task-chunk value
        _step_joint_config expects (see reach.py's own module docstring),
        so this helper's values map directly to real-world geometry."""
        chunk = np.zeros((horizon, 7))
        chunk[:, 0] = dx / CALVIN_ACTION_SCALE
        chunk[:, 1] = dy / CALVIN_ACTION_SCALE
        chunk[:, 2] = dz / CALVIN_ACTION_SCALE
        return chunk

    candidate = _xyz_chunk(0.02, 0.0, 0.0)
    other = _xyz_chunk(0.0, 0.1, 0.0)  # different axis -- clears the obstacle; 0.2/step now leaves JOINT_LIMITS
    # under the TCP-Jacobian predictor (checked), which would reject it for the wrong reason
    policy = FakeMultiCandidatePolicy([candidate, other])
    task_oracle = FakeTaskOracleWithTasks(success_after_steps=10**9, tasks={})

    far_away = Obstacle(center=np.array([100.0, 100.0, 100.0]), radius=0.02)
    shield = ArmSTLMonitorShield(obstacles=[far_away], epsilon=0.0)

    obstacle = Obstacle(center=_predicted_endpoint(Q_HOME, candidate), radius=0.05)

    result = run_calvin_shielded_subtask(
        env, policy, task_oracle, FakeLangEmbeddings(), SUBTASK, VAL_ANNOTATIONS, shield,
        ep_len=10, replan_steps=10, horizon_steps=10, obstacles=obstacle,
    )

    assert result["n_activated"] == 1  # candidate (candidates[0]) got rejected
    # `other`'s dx=0 -- FakeEnv's own dynamics (joint_angles[0] += action[0])
    # only ever moves index 0, so it staying exactly unchanged from Q_HOME
    # confirms `other` (not `candidate`, whose dx=0.02 would have moved it)
    # is what actually got executed.
    assert env.joint_angles[0] == Q_HOME[0]


class FakeRejectNominalCandidateShield:
    """Deterministically rejects candidates[0] -- the nominal chunk (best
    g(a); every score is 0 here) -- and executes candidates[1]. Isolates the
    REJECTED-CHUNK ground-truth wiring (does the harness run
    _privileged_rollout_violates on the rejected nominal chunk, and restore
    env state afterward) from real shield math (tests/test_arm_shield.py)."""

    def select(self, joint_angles, candidates, scores):
        del joint_angles, scores
        return candidates[1], {
            "fallback": False, "n_admissible": 1, "admissible_mask": [False, True],
        }


def test_shielded_subtask_a_rejected_candidate_that_truly_collides_counts_as_rejected_truly_unsafe():
    """rejected_total/rejected_truly_unsafe are measured by ACTUALLY
    stepping the rejected candidate through the real env (a privileged
    simulator rollout, matching the paper's own Sec. V.E definition, see
    _privileged_rollout_violates's own docstring) and restoring state
    afterward.

    Starts from Q_HOME (not np.zeros -- an all-zeros pose is degenerate:
    every joint but 0 folded at its own zero value means the WHOLE arm
    rotates as one rigid unit around joint 0's own axis, so an obstacle
    placed via that pose stays roughly equidistant across almost the
    entire rotation range -- verified directly, no direction "escapes" it.
    From Q_HOME, +0.1/step (unsafe_candidate) closes on the obstacle
    (clearance already negative after just 1 step) while -0.1/step clears
    it with growing margin -- a real, checked separation, not assumed.)
    `kept` (executed) stays at Q_HOME."""
    env = FakeEnv()
    env.reset()
    env.joint_angles = Q_HOME.copy()

    kept = _joint0_chunk(0.0)
    unsafe_candidate = _joint0_chunk(0.1)
    obstacle_joint_angles = Q_HOME.copy()
    obstacle_joint_angles[0] += 0.3
    obstacle = Obstacle(center=panda_frames(obstacle_joint_angles)[-1], radius=0.02)

    policy = FakeMultiCandidatePolicy([unsafe_candidate, kept])
    task_oracle = FakeTaskOracleWithTasks(success_after_steps=10**9, tasks={})
    shield = FakeRejectNominalCandidateShield()

    result = run_calvin_shielded_subtask(
        env, policy, task_oracle, FakeLangEmbeddings(), SUBTASK, VAL_ANNOTATIONS, shield,
        ep_len=10, replan_steps=10, horizon_steps=10, obstacles=obstacle,
    )

    assert result["n_activated"] == 1
    assert result["rejected_total"] == 1
    assert result["rejected_truly_unsafe"] == 1
    # env restored to exactly what `kept` (the real executed candidate,
    # all-zero delta) produces -- not left mid-way through testing the
    # rejected candidate's own privileged rollout.
    assert env.joint_angles[0] == Q_HOME[0]


def test_shielded_subtask_measure_intervention_precision_false_skips_privileged_rollout():
    """`measure_intervention_precision=False` must skip the real
    privileged-rollout ground-truth check entirely (see
    run_calvin_shielded_subtask's own docstring entry) -- rejected_total/
    rejected_truly_unsafe both stay 0 even though a candidate really was
    rejected here (n_activated still reflects that -- only the extra
    simulator check is skipped, not the cheap admissible-mask bookkeeping).
    Same setup as test_shielded_subtask_a_rejected_candidate_that_truly_
    collides_counts_as_rejected_truly_unsafe above, just with the flag off."""
    env = FakeEnv()
    env.reset()
    env.joint_angles = Q_HOME.copy()

    kept = _joint0_chunk(0.0)
    unsafe_candidate = _joint0_chunk(0.1)
    obstacle_joint_angles = Q_HOME.copy()
    obstacle_joint_angles[0] += 0.3
    obstacle = Obstacle(center=panda_frames(obstacle_joint_angles)[-1], radius=0.02)

    policy = FakeMultiCandidatePolicy([unsafe_candidate, kept])
    task_oracle = FakeTaskOracleWithTasks(success_after_steps=10**9, tasks={})
    shield = FakeRejectNominalCandidateShield()

    result = run_calvin_shielded_subtask(
        env, policy, task_oracle, FakeLangEmbeddings(), SUBTASK, VAL_ANNOTATIONS, shield,
        ep_len=10, replan_steps=10, horizon_steps=10, obstacles=obstacle,
        measure_intervention_precision=False,
    )

    assert result["n_activated"] == 1
    assert result["rejected_total"] == 0
    assert result["rejected_truly_unsafe"] == 0
    assert env.joint_angles[0] == Q_HOME[0]


def test_shielded_subtask_a_rejected_candidate_that_would_have_cleared_the_obstacle_is_not_counted_as_truly_unsafe():
    """Same setup, but `rejected_candidate` moves AWAY from the obstacle
    (-0.1/step from Q_HOME, verified to keep positive/growing clearance
    over the full 10-step horizon) -- a false-positive rejection (shield's
    own reasons for rejecting it are irrelevant here, isolated by
    FakeRejectNominalCandidateShield) must not be counted as
    rejected_truly_unsafe."""
    env = FakeEnv()
    env.reset()
    env.joint_angles = Q_HOME.copy()

    kept = _joint0_chunk(0.0)
    safe_candidate = _joint0_chunk(-0.1)  # moves away from the obstacle below
    obstacle_joint_angles = Q_HOME.copy()
    obstacle_joint_angles[0] += 0.3  # positive direction -- safe_candidate never goes there
    obstacle = Obstacle(center=panda_frames(obstacle_joint_angles)[-1], radius=0.02)

    policy = FakeMultiCandidatePolicy([safe_candidate, kept])
    task_oracle = FakeTaskOracleWithTasks(success_after_steps=10**9, tasks={})
    shield = FakeRejectNominalCandidateShield()

    result = run_calvin_shielded_subtask(
        env, policy, task_oracle, FakeLangEmbeddings(), SUBTASK, VAL_ANNOTATIONS, shield,
        ep_len=10, replan_steps=10, horizon_steps=10, obstacles=obstacle,
    )

    assert result["rejected_total"] == 1
    assert result["rejected_truly_unsafe"] == 0


def test_shielded_subtask_fallback_holds_the_gripper_instead_of_forcing_it_closed():
    """shield.select()'s fallback action is np.zeros_like(candidates[0])
    -- the harness must overwrite its gripper column with the real
    current gripper_action before stepping, or a fallback would (via
    HulcWrapper.step()'s real binarization, not exercised by this fake
    env) silently force the gripper closed regardless of its actual prior
    state. Checked here at the harness layer directly: whatever
    env.step() actually received.

    Also exercises the fallback window: a TOTAL fallback only commits 1
    env-step before re-proposing, not the full
    `replan_steps` window -- `FakeMultiCandidatePolicy` is stateless (same
    2 candidates every call, see its own docstring) and this threshold
    always disagrees, so EVERY decision here is a fallback -> with
    ep_len=10 and a 1-step window, that's 10 decisions, not 1."""
    env = FakeEnv()
    env.reset()
    env.gripper_action = 0.42  # distinctive marker, neither 0 nor +-1

    policy = FakeMultiCandidatePolicy([_joint0_chunk(0.1), _joint0_chunk(-0.1)])
    task_oracle = FakeTaskOracleWithTasks(success_after_steps=10**9, tasks={})
    shield = ArmConfThreshShield(disagreement_threshold=1e-9, horizon_steps=10)  # tiny -> both candidates disagree -> fallback

    result = run_calvin_shielded_subtask(
        env, policy, task_oracle, FakeLangEmbeddings(), SUBTASK, VAL_ANNOTATIONS, shield,
        ep_len=10, replan_steps=10, horizon_steps=10, obstacles=None, fallback_controller="freeze",
    )

    assert result["n_decisions"] == 10  # 1-step fallback window -> 10 decisions over ep_len=10
    assert result["n_activated"] == 10
    # Exactly one latency entry per decision (one per select() call).
    assert len(result["latencies_ms"]) == result["n_decisions"]
    assert result["n_fallback"] == 10
    assert np.isclose(env.last_gripper_action, 0.42)  # held (float32 roundtrip), not forced to 0/-1
    assert env.joint_angles[0] == 0.0  # froze in place -- fallback's position columns are 0


def test_shielded_subtask_retreat_fallback_pushes_away_from_the_obstacle():
    """fallback_controller='retreat' (opt-in; default is hold): on a TOTAL
    fallback, the executed action's position columns are no longer
    exactly 0 -- they push the nearest primitive away from the obstacle's
    center instead of freezing. Obstacle placed 5cm past the flange along
    +x with a large radius (0.2) so BOTH candidates (a small +-x joint-0
    nudge each) stay deeply inside it regardless of direction -- guarantees
    a total fallback every decision, same trick as the ConfThresh test
    above but via a real obstacle-aware shield instead of an impossible
    disagreement threshold."""
    env = FakeEnv()
    env.reset()
    env.joint_angles = Q_HOME.copy()

    policy = FakeMultiCandidatePolicy([_joint0_chunk(0.1), _joint0_chunk(-0.1)])
    task_oracle = FakeTaskOracleWithTasks(success_after_steps=10**9, tasks={})
    flange_position = panda_frames(Q_HOME)[FLANGE_FRAME_INDEX]
    obstacle = Obstacle(center=flange_position + np.array([0.05, 0.0, 0.0]), radius=0.2)
    shield = ArmSTLMonitorShield(obstacles=[obstacle], epsilon=0.0)

    result = run_calvin_shielded_subtask(
        env, policy, task_oracle, FakeLangEmbeddings(), SUBTASK, VAL_ANNOTATIONS, shield,
        ep_len=1, replan_steps=10, horizon_steps=10, obstacles=None, fallback_controller="retreat",
    )

    assert result["n_fallback"] == 1
    # FakeEnv's own dynamics only ever move joint_angles[0] via action[0]
    # (see its own docstring) -- a nonzero value here is direct evidence
    # the retreat delta's x-component was actually applied, not frozen.
    assert env.joint_angles[0] != 0.0


def test_shielded_subtask_retreat_fallback_degrades_to_freeze_without_obstacles():
    """ArmConfThreshShield has no `.obstacles` attribute at all (see its
    own docstring) -- fallback_controller='retreat' has nothing to push
    away from, so this must gracefully degrade to freeze (position
    columns stay exactly 0), not crash or guess a direction."""
    env = FakeEnv()
    env.reset()

    policy = FakeMultiCandidatePolicy([_joint0_chunk(0.1), _joint0_chunk(-0.1)])
    task_oracle = FakeTaskOracleWithTasks(success_after_steps=10**9, tasks={})
    shield = ArmConfThreshShield(disagreement_threshold=1e-9, horizon_steps=10)

    result = run_calvin_shielded_subtask(
        env, policy, task_oracle, FakeLangEmbeddings(), SUBTASK, VAL_ANNOTATIONS, shield,
        ep_len=1, replan_steps=10, horizon_steps=10, obstacles=None, fallback_controller="retreat",
    )

    assert result["n_fallback"] == 1
    assert env.joint_angles[0] == 0.0


class FakeAlwaysAdmissibleShield:
    """Plain shield with no `recertify`/`resolve` at all (like every real
    Arm*Shield) -- always executes candidates[0] unchanged. Used to pin
    the invariant that there is exactly one decision (one `select()`
    call) per `policy.propose()` call, for every baseline uniformly."""

    def select(self, joint_angles, candidates, scores):
        del joint_angles, scores
        mask = [True] * len(candidates)
        return candidates[0], {"fallback": False, "n_admissible": len(candidates), "admissible_mask": mask}


def test_shielded_subtask_n_decisions_equals_number_of_propose_calls():
    """Since there is no mid-window recertify()/resolve() mechanism,
    `n_decisions` must equal exactly the number of `policy.propose()`
    calls (`len(propose_latencies_ms)`) for EVERY shield, not just
    ArmConfThreshShield.
    `replan_steps=4 < horizon_steps=8 < chunk_length=10` (FakePolicy's own
    chunk length) exercises the general case, not the degenerate
    replan_steps==chunk_length one every other test in this file still
    uses. With `ep_len=10`: windows of 4, 4, 2 rows -> 3 propose() calls,
    3 decisions -- FakePolicy's own chunk length (10) never even matters
    here, since only `chunk[:replan_steps]` is ever executed per window."""
    env = FakeEnv()
    env.reset()
    policy = FakePolicy(delta=0.1, horizon=10)
    task_oracle = FakeTaskOracleWithTasks(success_after_steps=10**9, tasks={})
    shield = FakeAlwaysAdmissibleShield()

    result = run_calvin_shielded_subtask(
        env, policy, task_oracle, FakeLangEmbeddings(), SUBTASK, VAL_ANNOTATIONS, shield,
        ep_len=10, replan_steps=4, horizon_steps=8, obstacles=None,
    )

    assert len(result["propose_latencies_ms"]) == 3
    assert result["n_decisions"] == 3
    assert len(result["latencies_ms"]) == 3  # one latency entry per decision, i.e. per select() call
    assert np.isclose(env.joint_angles[0], 0.1 * 10)  # all 10 real steps still executed, across 3 decisions


def test_shielded_subtask_rejects_an_inconsistent_replan_horizon_pairing():
    """`0 < replan_steps <= horizon_steps <= CHUNK_LENGTH` is validated at
    the top of the harness (mirrors shortstop.calvin.config's own
    module-level assert) -- a caller passing `replan_steps > horizon_steps`
    (committing more rows than were ever certified) must fail loudly, not
    silently execute uncertified rows."""
    env = FakeEnv()
    env.reset()
    policy = FakePolicy(delta=0.1, horizon=10)
    task_oracle = FakeTaskOracleWithTasks(success_after_steps=10**9, tasks={})
    shield = FakeAlwaysAdmissibleShield()

    with pytest.raises(AssertionError):
        run_calvin_shielded_subtask(
            env, policy, task_oracle, FakeLangEmbeddings(), SUBTASK, VAL_ANNOTATIONS, shield,
            ep_len=10, replan_steps=8, horizon_steps=4, obstacles=None,
        )


class _FakeTargetPoseRobot:
    """calvin_env Robot stand-in with `use_target_pose`-style state:
    `target_pos`/`target_orn` are mutated IN PLACE (`+=`), exactly like
    Robot.relative_to_absolute, and are NOT part of serialize()."""

    def __init__(self):
        self.joints = np.zeros(3)
        self.target_pos = np.array([0.1, 0.2, 0.3])
        self.target_orn = np.array([0.0, 0.0, 1.0])

    def serialize(self):
        return {"joints": self.joints.copy()}

    def reset_from_storage(self, data):
        self.joints = data["joints"].copy()


class _FakeTargetPoseRawEnv:
    def __init__(self):
        self.robot = _FakeTargetPoseRobot()
        self.scene = _FakeSnapshotTarget(self, [])
        self.p = _FakeBulletClient()
        self.cid = 0

    def serialize(self):
        return {"robot": self.robot.serialize(), "scene": self.scene.serialize()}


class _FakeTargetPoseEnv:
    def __init__(self):
        self.env = _FakeTargetPoseRawEnv()


def test_snapshot_env_restores_in_place_mutated_robot_target_pose():
    """Regression: a bare env.env.serialize() misses the robot's
    accumulated IK target pose, so restoring left it wherever a forked
    branch had moved it (docs/calvin/env_verification.md). snapshot_env must copy it (not alias it -- the real robot
    mutates it with `+=`) and restore_env_snapshot must put it back."""
    env = _FakeTargetPoseEnv()
    robot = env.env.robot
    snapshot = snapshot_env(env)

    robot.joints += 1.0
    robot.target_pos += 0.5  # in place, like Robot.relative_to_absolute
    robot.target_orn += 0.5

    restore_env_snapshot(env, snapshot)

    np.testing.assert_array_equal(robot.joints, np.zeros(3))
    np.testing.assert_array_equal(robot.target_pos, [0.1, 0.2, 0.3])
    np.testing.assert_array_equal(robot.target_orn, [0.0, 0.0, 1.0])

    # A second fork from the same snapshot must also start clean.
    robot.target_pos += 0.5
    restore_env_snapshot(env, snapshot)
    np.testing.assert_array_equal(robot.target_pos, [0.1, 0.2, 0.3])


def test_delay_runs_the_old_tail_then_skips_the_new_chunks_first_rows():
    """d = 2, window 4: hold (no previous chunk) then B[2:4]; next decision
    B[4:6] then C[2:4]; a fallback runs the tail and leaves a holding one."""
    from shortstop.calvin.experiment import _Delay
    delay = _Delay(2)
    b = np.arange(10)[:, None] * np.ones((1, 7))
    c = 100 + b
    first = delay.rows(b, 4, False, gripper_action=1.0)
    assert np.all(first[:2, :6] == 0) and np.all(first[:2, 6] == 1.0)
    assert np.array_equal(first[2:], b[2:4])
    assert np.array_equal(delay.rows(c, 4, False, 1.0), np.concatenate([b[4:6], c[2:4]]))
    fallback = delay.rows(np.zeros((10, 7)), 1, True, -1.0)
    assert np.array_equal(fallback, c[4:6])
    held = delay.rows(b, 4, False, -1.0)
    assert np.all(held[:2, :6] == 0) and np.all(held[:2, 6] == -1.0) and np.array_equal(held[2:], b[2:4])
    assert np.array_equal(_Delay(0).rows(b, 4, False, 1.0), b[:4])


def test_delay_committed_rows_are_what_the_arm_runs_next():
    from shortstop.calvin.experiment import _Delay, _hold_row
    delay = _Delay(2)
    assert np.allclose(delay.committed(7, 1.0)[:, :6], 0.0)  # before the first chunk: still
    chunk = np.arange(70, dtype=float).reshape(10, 7)
    rows = delay.rows(chunk, 4, False, 1.0)
    assert np.allclose(rows[2:], chunk[2:4])
    committed = delay.committed(7, -1.0)
    assert np.allclose(committed[:, :6], chunk[4:6, :6]) and np.all(committed[:, 6] == -1.0)
    assert np.allclose(delay.rows(chunk, 4, False, 1.0)[:2], chunk[4:6])  # the rows the shield saw
    delay.rows(chunk, 1, True, 1.0, _hold_row(0.0))
    assert np.allclose(delay.committed(7, 1.0)[:, :6], 0.0)  # a hold tail is predicted as standing still
