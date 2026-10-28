"""Rollout harnesses for CALVIN.

- `run_calvin_unshielded_*`: Propose -> execute the policy's first candidate.
- `run_calvin_shielded_*`: Propose K candidates -> score them with g(a)
  (shortstop.calvin.progress) -> `shield.select` -> execute.

Env interface (mdt_policy's HulcWrapper): `env.step` takes a torch tensor of
7 relative actions; the raw 15-D proprioception is `obs["robot_obs_raw"]`
(ee_pos[0:3], ee_orn[3:6], gripper_width[6], joints[7:14], gripper_action[14]);
`task_oracle.get_task_info_for_set(start_info, current_info, {subtask})` is
CALVIN's own success check.

A subtask attempt stops at success, at the first collision with any obstacle
or at `ep_len`. `obstacles` is a list of spheres (None or [] = no obstacle). Ground-truth collision (`_clearance`) uses the 8 link
capsules plus the two real finger capsules at the current gripper width.
"""
import time

import numpy as np

from .config import CHUNK_LENGTH, FALLBACK_RETREAT_STEP, HORIZON_STEPS, POSE_MODEL, REPLAN_STEPS
from .progress import calvin_progress_scores, subtask_goals
from .reach import (
    CALVIN_ACTION_SCALE, CALVIN_ORIENTATION_SCALE, arm_step_capsules, arm_step_robustness, propagate_arm_tube,
)
from .robot_geometry import (
    FINGER_RADIUS,
    capsule_segments,
    closest_point_on_segment,
    finger_tip_capsules,
    panda_frames,
    point_to_segment_distance,
)

ROBOT_OBS_RAW_JOINT_SLICE = slice(7, 14)
ROBOT_OBS_RAW_GRIPPER_WIDTH_INDEX = 6
ROBOT_OBS_RAW_GRIPPER_INDEX = 14


def _robot_obs_raw(obs):
    return obs["robot_obs_raw"].detach().cpu().numpy()


def _joint_angles_from_obs(obs):
    return _robot_obs_raw(obs)[ROBOT_OBS_RAW_JOINT_SLICE]


def _gripper_width_from_obs(obs):
    return float(_robot_obs_raw(obs)[ROBOT_OBS_RAW_GRIPPER_WIDTH_INDEX])


def _gripper_action_from_obs(obs):
    return float(_robot_obs_raw(obs)[ROBOT_OBS_RAW_GRIPPER_INDEX])


def _base_transform_from_env(env):
    """(base_position, base_orientation) of the robot in the world; identity
    for test fakes without `env.env.robot`."""
    robot = getattr(getattr(env, "env", None), "robot", None)
    if robot is None:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)
    return robot.base_position, robot.base_orientation


def _to_action_tensor(action_row):
    import torch
    return torch.as_tensor(np.asarray(action_row, dtype=np.float32))


def _as_list(obstacles):
    """None / one Obstacle / a list -> a list."""
    if obstacles is None:
        return []
    return list(obstacles) if isinstance(obstacles, (list, tuple)) else [obstacles]


def _clearance_breakdown(obs, obstacles):
    """[(label, signed clearance)] for both fingers and every link capsule
    against every obstacle; [] without obstacles."""
    obstacles = _as_list(obstacles)
    if not obstacles:
        return []
    joint_angles = _joint_angles_from_obs(obs)
    (left_near, left_far), (right_near, right_far) = finger_tip_capsules(joint_angles, _gripper_width_from_obs(obs))
    primitives = [("finger_left", left_near, left_far, FINGER_RADIUS), ("finger_right", right_near, right_far, FINGER_RADIUS)]
    primitives += [(f"link_capsule[{i}] (frame{i}-frame{i+1})", a, b, r)
                   for i, (a, b, r) in enumerate(capsule_segments(joint_angles))]
    breakdown = []
    for k, obstacle in enumerate(obstacles):
        tag = "" if len(obstacles) == 1 else f" obstacle[{k}]"
        for label, a, b, r in primitives:
            d = point_to_segment_distance(obstacle.center, a, b)
            breakdown.append((label + tag, d - obstacle.radius - r))
    return breakdown


def _clearance(obs, obstacles):
    """Smallest surface-to-surface distance between the arm and any obstacle
    (<= 0 = collision), or None without obstacles."""
    breakdown = _clearance_breakdown(obs, obstacles)
    return min(value for _, value in breakdown) if breakdown else None


def snapshot_env(env):
    """`env.env.serialize()` plus a copy of the robot's IK target pose.

    CALVIN runs with `use_target_pose: true`: every relative action is added
    in place to `robot.target_pos/target_orn`, which IK then tracks. Its own
    serialize()/reset_from_storage() don't save that target, so a bare
    restore leaves the target where a forked branch put it and the arm
    drifts afterwards (up to ~0.1 m within a few steps)."""
    snapshot = env.env.serialize()
    robot = env.env.robot
    if hasattr(robot, "target_pos"):
        snapshot["robot_target_pose"] = (_copy_or_none(robot.target_pos), _copy_or_none(robot.target_orn))
    return snapshot


def _copy_or_none(value):
    return None if value is None else np.array(value, copy=True)


def restore_env_snapshot(env, snapshot, step=True):
    """Restore a `snapshot_env` state (a bare serialize() dict also works,
    without the target pose). `step`: one physics step afterwards (for
    rendering); off where the rollout continues from the restored state."""
    env.env.robot.reset_from_storage(snapshot["robot"])
    env.env.scene.reset_from_storage(snapshot["scene"])
    if "robot_target_pose" in snapshot:
        target_pos, target_orn = snapshot["robot_target_pose"]
        env.env.robot.target_pos = _copy_or_none(target_pos)
        env.env.robot.target_orn = _copy_or_none(target_orn)
    if step:
        env.env.p.stepSimulation(physicsClientId=env.env.cid)


def _privileged_rollout_violates(env, obstacles, chunk, snapshot=None):
    """Ground truth for intervention precision (paper Sec. V.E): execute a
    rejected `chunk` (the rows that would have run) in the real simulator,
    check for collision, then restore the state without stepping, so the
    measured rollout continues exactly as without the measurement. None
    without an obstacle. `snapshot`: reuse a state already captured."""
    if not _as_list(obstacles):
        return None
    if snapshot is None:
        snapshot = snapshot_env(env)
    violated = False
    for action_row in chunk:
        obs, _, _, _ = env.step(_to_action_tensor(action_row))
        if _clearance(obs, obstacles) <= 0:
            violated = True
            break
    restore_env_snapshot(env, snapshot, step=False)
    return violated


def _camera_frame(env):
    """(rgb_static, depth_static in meters) from one render."""
    camera_obs = env.env.get_obs()
    return camera_obs["rgb_obs"]["rgb_static"], camera_obs["depth_obs"]["depth_static"]


def _lang_goal(lang_embeddings, val_annotations, subtask):
    lang_annotation = val_annotations[subtask][0]
    goal = lang_embeddings.get_lang_goal(lang_annotation)
    goal["lang_text"] = lang_annotation
    return goal


class _Recorder:
    """Optional per-step recording for videos: arm frames + full env
    snapshots (`record_trajectory`) and camera images (`record_camera_frames`)."""

    def __init__(self, env, obs, record_trajectory, record_camera_frames):
        self.env = env
        self.trajectory = [] if record_trajectory else None
        self.state_snapshots = [] if record_trajectory else None
        self.camera_frames = [] if record_camera_frames else None
        self.depth_frames = [] if record_camera_frames else None
        self.add(obs)

    def add(self, obs):
        if self.trajectory is not None:
            self.trajectory.append(panda_frames(_joint_angles_from_obs(obs)))
            self.state_snapshots.append(snapshot_env(self.env))
        if self.camera_frames is not None:
            rgb, depth = _camera_frame(self.env)
            self.camera_frames.append(rgb)
            self.depth_frames.append(depth)

    def into(self, result, obstacles):
        if self.trajectory is not None:
            result["trajectory"] = self.trajectory
            result["state_snapshots"] = self.state_snapshots
        if self.trajectory is not None or self.camera_frames is not None:
            result["obstacles"] = obstacles
        if self.camera_frames is not None:
            result["camera_frames"] = self.camera_frames
            result["depth_frames"] = self.depth_frames
        return result


def _hold_row(gripper_action):
    """Marker row for one joint-hold step (`_hold_step`): NaN position columns."""
    row = np.full(7, np.nan)
    row[6] = gripper_action
    return row


def _hold_step(env, gripper_action, anchor=None):
    """One env step that holds the arm at joint angles `anchor` (default: the
    current ones) with CALVIN's "joint_abs" action, then puts the Cartesian IK
    target on the reached TCP pose so the next relative action continues from
    where the arm is. Returns what env.step returns."""
    robot = env.env.robot
    _state, info = robot.get_observation()
    if anchor is None:
        anchor = info["arm_joint_states"]
    o, r, d, i = env.env.step({"type": "joint_abs", "action": [*anchor, 1 if gripper_action > 0 else -1]})
    _state, info = robot.get_observation()
    robot.target_pos = np.array(info["tcp_pos"])
    robot.target_orn = np.array(info["tcp_orn"])
    return env.transform_observation(o), r, d, i


class _Outcome:
    """Executes action rows and tracks success / collision / step budget."""

    def __init__(self, env, task_oracle, subtask, start_info, obstacles, ep_len, recorder):
        self.env, self.task_oracle, self.subtask, self.start_info = env, task_oracle, subtask, start_info
        self.obstacles, self.ep_len, self.recorder = obstacles, ep_len, recorder
        self.violated = self.reached = False
        self.min_clearance = None
        self.steps_taken = 0
        self.executed_joint_angles = []
        self.hold_anchor = None

    @property
    def done(self):
        return self.violated or self.reached or self.steps_taken >= self.ep_len

    def execute(self, rows):
        """Step through `rows`, stopping early when done; returns the last obs."""
        obs = None
        for action_row in rows:
            if np.isnan(action_row[0]):
                # Consecutive hold steps keep the angles of the first one: no creep.
                if self.hold_anchor is None:
                    self.hold_anchor = list(self.env.env.robot.get_observation()[1]["arm_joint_states"])
                obs, _, _, current_info = _hold_step(self.env, action_row[6], self.hold_anchor)
            else:
                self.hold_anchor = None
                obs, _, _, current_info = self.env.step(_to_action_tensor(action_row))
            self.steps_taken += 1
            self.executed_joint_angles.append(_joint_angles_from_obs(obs))
            self.recorder.add(obs)
            clearance = _clearance(obs, self.obstacles)
            if clearance is not None:
                self.min_clearance = clearance if self.min_clearance is None else min(self.min_clearance, clearance)
                if clearance <= 0:
                    self.violated = True
                    break
            if self.task_oracle.get_task_info_for_set(self.start_info, current_info, {self.subtask}):
                self.reached = True
                break
            if self.steps_taken >= self.ep_len:
                break
        return obs

    def result(self):
        return {"violated": self.violated, "reached": self.reached, "min_clearance": self.min_clearance,
                "steps_taken": self.steps_taken}


class _Delay:
    """Inference delay of `steps` env steps (Real-Time Chunking, Black et al.
    2025): a chunk computed at step t arrives at t + d, the arm runs the
    previous chunk's next d rows meanwhile, and the new chunk's first d rows
    (their time has passed) are skipped. Each decision still spans its
    window: d old rows + rows d..window-1 of the new chunk; the new chunk's
    rows window..window+d-1 become the next tail. Before the first chunk and
    after a fallback the tail holds the arm still. steps = 0: no delay."""

    def __init__(self, steps):
        self.steps = int(steps)
        self.tail = None

    def rows(self, chunk, window, fallback, gripper_action, hold_row=None):
        if self.steps == 0:
            return chunk[:window]
        hold = np.zeros((self.steps, chunk.shape[1])) if hold_row is None else np.tile(hold_row, (self.steps, 1))
        hold[:, 6] = gripper_action
        old = hold if self.tail is None else self.tail
        rows = np.concatenate([old, chunk[self.steps:window]])
        self.tail = hold if fallback else np.asarray(chunk[window:window + self.steps])
        return rows

    def committed(self, width, gripper_action):
        """The `steps` rows the arm runs before the next chunk can act, as the
        shield predicts them (a hold row = standing still)."""
        rows = np.zeros((self.steps, width)) if self.tail is None else np.array(self.tail, dtype=float)
        rows = np.nan_to_num(rows, nan=0.0)
        rows[:, 6] = gripper_action
        return rows


def run_calvin_unshielded_subtask(
    env, policy, task_oracle, lang_embeddings, subtask, val_annotations,
    ep_len=360, replan_steps=REPLAN_STEPS, obstacles=None, record_trajectory=False, record_camera_frames=False,
    delay_steps=0,
):
    """One subtask attempt executing `candidates[0]` of every proposal
    (`delay_steps`: inference delay, see `_Delay`).

    Returns {violated, reached, min_clearance (None without an obstacle),
    steps_taken, propose_latencies_ms}, plus the recordings when asked."""
    obs = env.get_obs()
    goal = _lang_goal(lang_embeddings, val_annotations, subtask)
    recorder = _Recorder(env, obs, record_trajectory, record_camera_frames)
    obstacles = _as_list(obstacles)
    outcome = _Outcome(env, task_oracle, subtask, env.get_info(), obstacles, ep_len, recorder)
    propose_latencies_ms = []
    delay = _Delay(delay_steps)
    while not outcome.done:
        t0 = time.perf_counter()
        candidates = policy.propose({**obs, "goal": goal})
        propose_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        chunk = np.asarray(candidates[0])
        obs = outcome.execute(delay.rows(chunk, replan_steps, False, _gripper_action_from_obs(obs)))
    result = {**outcome.result(), "propose_latencies_ms": propose_latencies_ms}
    return recorder.into(result, obstacles)


def _reset(env, initial_condition, get_env_state_for_initial_condition):
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_condition)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)


def run_calvin_unshielded_sequence(
    env, policy, task_oracle, lang_embeddings, initial_condition, eval_sequence, val_annotations,
    get_env_state_for_initial_condition, ep_len=360, replan_steps=REPLAN_STEPS, obstacles=None,
    record_trajectory=False, record_camera_frames=False, delay_steps=0,
):
    """Reset to `initial_condition`, then attempt `eval_sequence`'s subtasks
    in order, stopping at the first one not reached (CALVIN's own
    truncation). Returns the list of per-subtask results."""
    _reset(env, initial_condition, get_env_state_for_initial_condition)
    attempts = []
    for subtask in eval_sequence:
        result = run_calvin_unshielded_subtask(
            env, policy, task_oracle, lang_embeddings, subtask, val_annotations, ep_len=ep_len,
            replan_steps=replan_steps, obstacles=obstacles, record_trajectory=record_trajectory,
            record_camera_frames=record_camera_frames, delay_steps=delay_steps,
        )
        attempts.append(result)
        if not result["reached"]:
            break
    return attempts


def _brake_action(env, obs):
    """(6,) action that moves CALVIN's IK target onto the current TCP pose: a
    stop command (the arm otherwise keeps converging to a target it trails).
    Zeros for test fakes without `env.env.robot`."""
    robot = getattr(getattr(env, "env", None), "robot", None)
    if robot is None or getattr(robot, "target_pos", None) is None:
        return np.zeros(6)
    raw = _robot_obs_raw(obs)
    d_orn = (raw[3:6] - np.asarray(robot.target_orn) + np.pi) % (2 * np.pi) - np.pi
    return np.concatenate([(raw[0:3] - np.asarray(robot.target_pos)) / CALVIN_ACTION_SCALE,
                           d_orn / CALVIN_ORIENTATION_SCALE])


def _retreat_fallback_delta(obs, obstacles, retreat_step):
    """Raw (3,) action that pushes the arm primitive nearest to any obstacle
    straight away from it by `retreat_step` meters -- a one-shot reflex on the
    real geometry, not a certified repair. Zeros without obstacles."""
    if not obstacles:
        return np.zeros(3)
    joint_angles = _joint_angles_from_obs(obs)
    primitives = [(near, far, FINGER_RADIUS) for near, far in finger_tip_capsules(joint_angles, _gripper_width_from_obs(obs))]
    primitives += list(capsule_segments(joint_angles))
    best_gap, best_point, best_obstacle = None, None, None
    for obstacle in obstacles:
        for point_a, point_b, radius in primitives:
            point = closest_point_on_segment(obstacle.center, point_a, point_b)
            gap = float(np.linalg.norm(point - obstacle.center)) - obstacle.radius - radius
            if best_gap is None or gap < best_gap:
                best_gap, best_point, best_obstacle = gap, point, obstacle
    direction = best_point - best_obstacle.center
    norm = np.linalg.norm(direction)
    unit = direction / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])
    return unit * (retreat_step / CALVIN_ACTION_SCALE)


def run_calvin_shielded_subtask(
    env, policy, task_oracle, lang_embeddings, subtask, val_annotations, shield,
    ep_len=360, replan_steps=REPLAN_STEPS, horizon_steps=HORIZON_STEPS, obstacles=None,
    record_trajectory=False, record_camera_frames=False, fallback_controller="hold",
    retreat_step=FALLBACK_RETREAT_STEP, freeze_steps=None, measure_intervention_precision=True, use_progress=True,
    log_soundness=False, delay_steps=0, delay_aware=False,
):
    """One subtask attempt with a shield: every decision proposes K
    candidates, scores them with g(a), calls `shield.select` once and
    executes `replan_steps` rows of the result (the shield certifies its own
    `horizon_steps` window).

    - `obstacles`, when given, replace an obstacle-aware shield's `obstacles`.
    - `use_progress=False` scores every candidate 0, so Select keeps the
      policy's own order (the no-g(a) arm of a comparison).
    - Total fallback (no admissible candidate): the shield returns zeros; the
      gripper column is set to the current gripper action (HulcWrapper
      binarizes 0 to "close"); "hold" (default; the paper's braking controller, no
      obstacle information) holds the joint angles at the start of the stall (`_hold_step`);
      "brake" moves CALVIN's IK target onto the current TCP pose
      (`_brake_action`; the arm still drifts in its null space); "freeze" keeps the
      zeros (the arm keeps converging to its old target); "retreat" moves the arm `retreat_step` m
      away from the nearest obstacle (`_retreat_fallback_delta`, a
      hand-designed geometric reflex, used as an ablation); only 1 row (or
      `freeze_steps`) is executed before proposing again -- a fresh sample
      is the way out of a state where everything was rejected.
    - `measure_intervention_precision`: run every rejected candidate in the
      real simulator (`_privileged_rollout_violates`); costs extra sim steps.
    - `log_soundness` (shields with a reachtube): per executed row of a
      non-fallback decision, [decision, step k, certified robustness of tube
      step k, true robustness of the arm reached (same primitives, no
      inflation), max frame deviation predicted vs reached] -- passive.
    - `delay_steps`: inference delay (`_Delay`); the shield does not know
      the delay unless `delay_aware`: then it certifies what will run, the
      committed rows followed by rows d.. of each candidate, over
      H = replan_steps + d (Theorem 1 must cover every row executed before
      the next certificate), and never modifies the committed rows.

    Returns the unshielded fields plus n_decisions, n_activated (decisions
    whose nominal chunk was rejected, see below), n_fallback, latencies_ms (select), progress_latencies_ms
    (g(a)), rejected_total, rejected_truly_unsafe, status_counts (per-decision
    shield status: MPC/CBF solver status, ShortStop repair rounds),
    decision_fallback (per decision), g_a_has_goal and g_a_phase2_step (step
    at which g(a) switched to the completion goal, or None)."""
    assert 0 < replan_steps <= horizon_steps <= CHUNK_LENGTH, (
        f"need 0 < replan_steps={replan_steps} <= horizon_steps={horizon_steps} <= CHUNK_LENGTH={CHUNK_LENGTH}"
    )
    obs = env.get_obs()
    goal = _lang_goal(lang_embeddings, val_annotations, subtask)
    start_info = env.get_info()
    base_position, base_orientation = _base_transform_from_env(env)
    goals = subtask_goals(task_oracle, subtask, start_info, _joint_angles_from_obs(obs),
                          base_position, base_orientation) if use_progress else None
    obstacles = _as_list(obstacles)
    if obstacles and hasattr(shield, "obstacles"):
        shield.obstacles = obstacles

    recorder = _Recorder(env, obs, record_trajectory, record_camera_frames)
    outcome = _Outcome(env, task_oracle, subtask, start_info, obstacles, ep_len, recorder)
    stats = {"n_decisions": 0, "n_activated": 0, "n_fallback": 0, "rejected_total": 0, "rejected_truly_unsafe": 0}
    latencies_ms, propose_latencies_ms, progress_latencies_ms = [], [], []
    g_a_phase2_step = None
    status_counts = {}
    decision_fallback = []
    soundness = []
    disagreement = []
    delay = _Delay(delay_steps)
    aware = delay_aware and delay_steps > 0
    if aware:
        shield.fixed_prefix = delay_steps
        for attr in ("horizon_steps", "steps"):
            if hasattr(shield, attr):
                setattr(shield, attr, replan_steps + delay_steps)

    while not outcome.done:
        joint_angles = _joint_angles_from_obs(obs)
        tcp_euler = _robot_obs_raw(obs)[3:6] if POSE_MODEL else None
        if hasattr(shield, "tcp_euler"):
            shield.tcp_euler = tcp_euler
        t0 = time.perf_counter()
        candidates = policy.propose({**obs, "goal": goal})
        propose_latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        goal_box = None if goals is None else goals.target(env.get_info()["scene_info"])
        scores = calvin_progress_scores(joint_angles, candidates, goal_box, tcp_euler)
        progress_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        if goals is not None and goals.phase == 2 and g_a_phase2_step is None:
            g_a_phase2_step = outcome.steps_taken

        shield_candidates = candidates
        if aware:
            committed = delay.committed(np.asarray(candidates[0]).shape[1], _gripper_action_from_obs(obs))
            shield_candidates = [np.concatenate([committed, np.asarray(c)[delay_steps:]]) for c in candidates]
        t0 = time.perf_counter()
        chunk, shield_info = shield.select(joint_angles, shield_candidates, scores)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        stats["n_decisions"] += 1
        if "status" in shield_info:
            status_counts[shield_info["status"]] = status_counts.get(shield_info["status"], 0) + 1
        if "disagreement" in shield_info:
            disagreement.append(shield_info["disagreement"])
        # Activation (paper Sec. V.E: the shield rejected/repaired/fell back) is
        # judged on the chunk the method would execute without a shield: the
        # best-g(a) candidate for selection shields, candidates[0] for the
        # correcting ones (MPC-Filter/CBF). Same for every baseline, so one
        # rejected chunk per activated decision; other rejected candidates that
        # would never have run do not count. Precision rolls that chunk out for
        # the rows that would have executed (replan_steps).
        nominal = 0 if getattr(shield, "corrects_first_candidate", False) else int(np.argmax(scores))
        accepted = shield_info.get("certified_mask", shield_info["admissible_mask"])
        decision_fallback.append(bool(shield_info["fallback"]))
        if shield_info["fallback"] or not accepted[nominal]:
            stats["n_activated"] += 1
            if measure_intervention_precision:
                stats["rejected_total"] += 1
                stats["rejected_truly_unsafe"] += bool(
                    _privileged_rollout_violates(env, obstacles, candidates[nominal][:replan_steps]))

        if shield_info["fallback"]:
            stats["n_fallback"] += 1
            if fallback_controller == "hold":
                chunk[0] = _hold_row(_gripper_action_from_obs(obs))
            elif fallback_controller == "brake":
                chunk[0, :6] = _brake_action(env, obs)
            elif fallback_controller == "retreat":
                chunk[0, :3] = _retreat_fallback_delta(obs, getattr(shield, "obstacles", None), retreat_step)
            chunk[:, 6] = _gripper_action_from_obs(obs)
            window = freeze_steps if freeze_steps is not None else 1
        else:
            window = replan_steps
        if aware and not shield_info["fallback"]:  # back to the new chunk's own row indices
            chunk = np.concatenate([np.zeros((delay_steps, chunk.shape[1])), np.asarray(chunk)[delay_steps:]])
        tube = None
        if log_soundness and not delay_steps and not shield_info["fallback"] and hasattr(shield, "model_error"):
            tube = propagate_arm_tube(joint_angles, chunk[:window], shield.w_bar, shield.model_error,
                                      tcp_euler=tcp_euler)
            first = len(outcome.executed_joint_angles)
        obs = outcome.execute(delay.rows(
            np.asarray(chunk), window, shield_info["fallback"], _gripper_action_from_obs(obs),
            _hold_row(0.0) if fallback_controller == "hold" and shield_info["fallback"] else None))
        if tube is not None:
            for k, q in enumerate(outcome.executed_joint_angles[first:], 1):
                actual = arm_step_capsules(q)
                deviation = max(float(np.linalg.norm(actual[i].a - tube[k][i].a)) for i in range(9))
                soundness.append([stats["n_decisions"] - 1, k, arm_step_robustness(tube[k], obstacles),
                                  arm_step_robustness(actual, obstacles), deviation])

    result = {
        **outcome.result(), **stats,
        "latencies_ms": latencies_ms, "propose_latencies_ms": propose_latencies_ms,
        "progress_latencies_ms": progress_latencies_ms,
        "g_a_has_goal": goals is not None, "g_a_phase2_step": g_a_phase2_step,
        "status_counts": status_counts, "decision_fallback": decision_fallback,
    }
    if log_soundness:
        result["soundness"] = soundness
    if disagreement:
        result["disagreement"] = disagreement
    return recorder.into(result, obstacles)


def run_calvin_shielded_sequence(
    env, policy, task_oracle, lang_embeddings, initial_condition, eval_sequence, val_annotations,
    get_env_state_for_initial_condition, shield, ep_len=360, replan_steps=REPLAN_STEPS,
    horizon_steps=HORIZON_STEPS, obstacles=None, record_trajectory=False, record_camera_frames=False,
    fallback_controller="hold", retreat_step=FALLBACK_RETREAT_STEP, freeze_steps=None,
    measure_intervention_precision=True, use_progress=True, log_soundness=False, delay_steps=0, delay_aware=False,
):
    """Shielded counterpart of `run_calvin_unshielded_sequence`."""
    _reset(env, initial_condition, get_env_state_for_initial_condition)
    attempts = []
    for subtask in eval_sequence:
        result = run_calvin_shielded_subtask(
            env, policy, task_oracle, lang_embeddings, subtask, val_annotations, shield,
            ep_len=ep_len, replan_steps=replan_steps, horizon_steps=horizon_steps, obstacles=obstacles,
            record_trajectory=record_trajectory, record_camera_frames=record_camera_frames,
            fallback_controller=fallback_controller, retreat_step=retreat_step, freeze_steps=freeze_steps,
            measure_intervention_precision=measure_intervention_precision, use_progress=use_progress,
            log_soundness=log_soundness, delay_steps=delay_steps, delay_aware=delay_aware,
        )
        attempts.append(result)
        if not result["reached"]:
            break
    return attempts
