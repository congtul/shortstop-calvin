"""Shields for the Panda arm on CALVIN (Certify / Select / Repair over the
reachtube of shortstop.calvin.reach), plus the Table 1 baselines.

| class               | role                                                     |
|---------------------|----------------------------------------------------------|
| ArmReachOnlyShield  | reject any candidate whose tube touches an obstacle      |
| ArmSTLShield        | reject if robustness-to-go (Eq. 2) < epsilon             |
| ArmSTLMonitorShield | STL-Monitor: ArmSTLShield on the nominal path (no tube)  |
| ArmRepairShield     | ShortStop: STL certify + counterexample-guided repair    |
| ArmMPCFilterShield  | MPC-Filter: H-step SQP correcting candidates[0]          |
| ArmCBFShield        | CBF-Shield: CBF QP on each executed action, in turn      |
| ArmConfThreshShield | Conf-Thresh: reject on sampler-ensemble disagreement     |

The ranking shields pick the admissible candidate with the best caller
supplied score g(a) (shortstop.calvin.progress). Every shield also rejects a
candidate whose predicted joint trajectory leaves JOINT_LIMITS. Actions map
to joint motion through the TCP Jacobian (reach._step_joint_config).
"""
import cvxpy as cp
import numpy as np

from ..common.shield_base import ShieldBase
from .reach import (
    CALVIN_ACTION_SCALE, arm_find_counterexample, arm_robustness_to_go, model_error_at, nominal_joint_trajectory,
    pose_step_parts, position_joint_map, propagate_arm_tube, step_joint_config,
)
from .config import JOINT_ERROR, MODEL_ERROR, SAFETY_MARGIN
from .robot_geometry import (
    FRAME_RADIUS, GRIPPER_TIP_RADIUS, JOINT_LIMITS, LINK_RADIUS, N_JOINTS,
    all_frame_jacobians, gripper_tip_jacobian, gripper_tip_position, numerical_jacobian,
    panda_frames, within_joint_limits,
)


# A QP correction smaller than this (raw action units; 1e-3 = 20 um of TCP
# motion) is solver tolerance, not an intervention.
QP_INTERVENTION_TOL = 1e-3

# MPC-Filter's QP keeps this much (m) beyond the constraint boundary, so that
# SQP iterates land strictly feasible instead of converging onto it.
SQP_MARGIN = 1e-3


def _closest_point_t(point, a, b):
    """Parameter t in [0, 1] of the closest point on segment ab to `point`
    (a link's closest point and its Jacobian are the same blend of its two
    end frames)."""
    ab = b - a
    length_sq = float(ab @ ab)
    if length_sq < 1e-12:
        return 0.0
    return float(np.clip((point - a) @ ab / length_sq, 0.0, 1.0))


def _arm_geometry(q):
    """(frame points, frame Jacobians, TCP, TCP Jacobian) at joint config q."""
    positions, jacobians = all_frame_jacobians(q)
    return positions, jacobians, gripper_tip_position(q), gripper_tip_jacobian(q)


def _iter_primitives_for_obstacle(geometry, obstacle):
    """(point, Jacobian, physical radius) for the reachtube's primitives
    (reach.propagate_arm_tube): the 9 frames (the flange with its link
    radius), the point of each of the 8 links and of the flange->TCP segment
    closest to `obstacle` (Jacobian blended between the segment's ends)."""
    frame_points, frame_jacobians, tip, tip_jacobian = geometry
    flange = len(frame_points) - 1
    for i, point in enumerate(frame_points):
        yield point, frame_jacobians[i], LINK_RADIUS[-1] if i == flange else FRAME_RADIUS[i]
    t = _closest_point_t(obstacle.center, frame_points[flange], tip)
    yield (frame_points[flange] + t * (tip - frame_points[flange]),
           (1.0 - t) * frame_jacobians[flange] + t * tip_jacobian, GRIPPER_TIP_RADIUS)
    for i in range(len(LINK_RADIUS)):
        a_pt, b_pt = frame_points[i], frame_points[i + 1]
        t = _closest_point_t(obstacle.center, a_pt, b_pt)
        link_point = a_pt + t * (b_pt - a_pt)
        link_jacobian = (1.0 - t) * frame_jacobians[i] + t * frame_jacobians[i + 1]
        yield link_point, link_jacobian, LINK_RADIUS[i]


def _solve_or_none(problem, var):
    """(solution or None, status): status is cvxpy's problem status, or
    "solver_error" when the solver raised."""
    try:
        problem.solve()
    except cp.error.SolverError:
        return None, "solver_error"
    if var.value is None or problem.status not in ("optimal", "optimal_inaccurate"):
        return None, str(problem.status)
    return np.asarray(var.value), str(problem.status)


class ArmReachOnlyShield(ShieldBase):
    """Reject any candidate whose reachtube touches an obstacle.

    `horizon_steps`: rows certified (None = the whole chunk)."""

    def __init__(self, obstacles, w_bar, model_error=MODEL_ERROR, horizon_steps=None, joint_error=JOINT_ERROR):
        self.obstacles = obstacles
        self.w_bar = w_bar
        self.model_error = model_error
        self.horizon_steps = horizon_steps
        self.joint_error = joint_error
        # Certified clearance kept from every obstacle (paper: X_s = {dist >= eps});
        # covers the arm's drift under the brake fallback (Assumption 3).
        self.safety_margin = SAFETY_MARGIN
        # Current TCP orientation (Euler), set by the harness every decision:
        # selects the pose model of f-hat; None = position-only model.
        self.tcp_euler = None

    # Rows already committed to the arm (inference delay, `delay_aware`): they
    # are certified as part of the chunk but never modified.
    fixed_prefix = 0

    def _joint_limits(self, k):
        """JOINT_LIMITS shrunk by the step-k joint prediction error (rad)."""
        margin = model_error_at(self.joint_error, k)
        return JOINT_LIMITS[:, 0] + margin, JOINT_LIMITS[:, 1] - margin

    def _trajectory_within_joint_limits(self, joint_angles, task_chunk):
        """Does the predicted joint trajectory, widened by the joint error,
        stay inside JOINT_LIMITS?"""
        trajectory = nominal_joint_trajectory(joint_angles, task_chunk, self.tcp_euler)
        return all(np.all(q >= lo) and np.all(q <= hi)
                   for k, q in enumerate(trajectory[1:], 1) for lo, hi in [self._joint_limits(k)])

    def _admissible(self, joint_angles, task_chunk):
        task_chunk = task_chunk[:self.horizon_steps]
        if not self._trajectory_within_joint_limits(joint_angles, task_chunk):
            return False
        tube = propagate_arm_tube(joint_angles, task_chunk, self.w_bar, self.model_error, tcp_euler=self.tcp_euler)
        return arm_robustness_to_go(tube, self.obstacles) >= self.safety_margin


class ArmSTLShield(ArmReachOnlyShield):
    """Reject if the tube's robustness-to-go (Eq. 2) is below `epsilon`."""

    def __init__(self, obstacles, w_bar, model_error=MODEL_ERROR, epsilon=0.02, horizon_steps=None):
        super().__init__(obstacles, w_bar, model_error, horizon_steps)
        self.epsilon = epsilon

    def _admissible(self, joint_angles, task_chunk):
        task_chunk = task_chunk[:self.horizon_steps]
        if not self._trajectory_within_joint_limits(joint_angles, task_chunk):
            return False
        tube = propagate_arm_tube(joint_angles, task_chunk, self.w_bar, self.model_error, tcp_euler=self.tcp_euler)
        return arm_robustness_to_go(tube, self.obstacles) >= self.epsilon


class ArmSTLMonitorShield(ArmSTLShield):
    """STL-Monitor (paper Sec. V.D): nominal STL robustness, no reachtube
    inflation (w_bar = model_error = 0; the arm's physical radii stay), no
    repair. `epsilon` must be given: the paper says both "rejects if
    negative" (0) and "identical epsilon for all model-based baselines"."""

    def __init__(self, obstacles, epsilon, horizon_steps=None):
        super().__init__(obstacles, w_bar=0.0, model_error=0.0, epsilon=epsilon, horizon_steps=horizon_steps)
        self.joint_error = 0.0


class ArmRepairShield(ArmSTLShield):
    """ShortStop (Algorithm 1): certify each candidate's first `horizon_steps`
    rows. Repair runs only when none certifies: failing candidates are
    repaired in g(a) order and the first certified repair is executed.
    "gradient" repair: up to `max_repair_iters` (paper: 1) steps along the
    counterexample direction (Eq. 3-4), each limited to `trust_region` from
    the original and re-certified (`step_size`, `trust_region` in meters of
    TCP motion). "polyak" / "ce_qp": see _repair_linearized; at most
    `max_linearized_iters` re-linearizations per candidate and `repair_budget`
    per decision (all candidates), the real-time bound."""

    def __init__(
        self, obstacles, w_bar, model_error=MODEL_ERROR, epsilon=0.02,
        trust_region=0.05, step_size=0.02, max_repair_iters=1, horizon_steps=None,
        repair_mode="gradient", repair_margin=0.003, repair_row_cap=0.10, max_linearized_iters=20,
        repair_budget=None,
    ):
        super().__init__(obstacles, w_bar, model_error, epsilon, horizon_steps)
        self.trust_region = trust_region
        self.step_size = step_size
        self.max_repair_iters = max_repair_iters
        # "gradient": the paper's single step a + eta d inside an l2 trust region.
        # "polyak" / "ce_qp": minimum-norm corrections of the linearized violated
        # constraints (see _repair_linearized); trust_region/step_size unused,
        # repair_row_cap (m of TCP motion per row) only guards against runaway.
        if repair_mode not in ("gradient", "polyak", "ce_qp"):
            raise ValueError(f"repair_mode={repair_mode!r}")
        self.repair_mode = repair_mode
        self.repair_margin = repair_margin
        self.repair_row_cap = repair_row_cap
        self.max_linearized_iters = max_linearized_iters
        self.repair_budget = repair_budget  # re-linearizations per decision, over all candidates
        self._rounds = 0

    def _certify(self, joint_angles, task_chunk):
        """(tube, ok): ok = within joint limits and robustness-to-go >= epsilon."""
        task_chunk = task_chunk[:self.horizon_steps]
        tube = propagate_arm_tube(joint_angles, task_chunk, self.w_bar, self.model_error, tcp_euler=self.tcp_euler)
        joint_limits_ok = self._trajectory_within_joint_limits(joint_angles, task_chunk)
        ok = joint_limits_ok and arm_robustness_to_go(tube, self.obstacles) >= self.epsilon
        return tube, ok

    def _witness_jacobian(self, frame_key, q_at_witness_step, obstacle_center):
        """(3, 7) Jacobian of the counterexample witness point: a frame's own,
    or the blend of the two end frames' for a link / the fingertip capsule."""
        frames = panda_frames(q_at_witness_step)
        if isinstance(frame_key, tuple) and frame_key[0] == "link":
            i = frame_key[1]
            t = _closest_point_t(obstacle_center, frames[i], frames[i + 1])
            return (1.0 - t) * numerical_jacobian(q_at_witness_step, i) + t * numerical_jacobian(q_at_witness_step, i + 1)
        if frame_key == "fingertip":
            tip = gripper_tip_position(q_at_witness_step)
            t = _closest_point_t(obstacle_center, frames[-1], tip)
            flange_index = len(frames) - 1
            return (
                (1.0 - t) * numerical_jacobian(q_at_witness_step, flange_index)
                + t * gripper_tip_jacobian(q_at_witness_step)
            )
        return numerical_jacobian(q_at_witness_step, frame_key)  # plain frame index (int)

    def _repair_direction(self, joint_angles, task_chunk, counterexample):
        """Unit action-space direction increasing the witness's distance to the
    obstacle (Eq. 4): the Cartesian normal n mapped through
    M = J_witness @ pinv(J_tcp) as M.T @ n (exactly n when the witness is
    the TCP); falls back to n near singular poses."""
        ce = counterexample
        direction = ce["witness"] - ce["obstacle"].center
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            n = np.array([0.0, 0.0, 1.0])  # witness at the obstacle center: push up
        else:
            n = direction / norm

        q_at_witness_step = nominal_joint_trajectory(joint_angles, task_chunk, self.tcp_euler)[ce["step"]]
        witness_jacobian = self._witness_jacobian(ce["frame"], q_at_witness_step, ce["obstacle"].center)
        m = witness_jacobian @ position_joint_map(q_at_witness_step, self.tcp_euler is not None)  # dp_witness ~= m @ da
        projected = m.T @ n  # d(dist)/d(da) = n @ m = m.T @ n
        projected_norm = np.linalg.norm(projected)
        if projected_norm < 1e-9:
            return n
        return projected / projected_norm

    def _violations(self, joint_angles, chunk, tube, worst_only=False):
        """[(k, rho_k, g_k)] for every step (or only the worst one) whose worst
        capsule-obstacle distance is below epsilon + repair_margin, with the
        linearization rho_k(a + D) ~ rho_k + sum_{j<=k} g_k . D_j (D_j in raw
        action units)."""
        q_traj = nominal_joint_trajectory(joint_angles, chunk[:self.horizon_steps], self.tcp_euler)
        found = []
        for k in range(1, len(tube)):
            ce = arm_find_counterexample([tube[0], tube[k]], self.obstacles)
            if ce is not None and ce["robustness"] < self.epsilon + self.repair_margin:
                found.append((k, ce))
        if worst_only and found:
            found = [min(found, key=lambda kc: kc[1]["robustness"])]
        out = []
        for k, ce in found:
            normal = ce["witness"] - ce["obstacle"].center
            norm = np.linalg.norm(normal)
            normal = normal / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])
            q_k = q_traj[k]
            m = (self._witness_jacobian(ce["frame"], q_k, ce["obstacle"].center)
                 @ position_joint_map(q_k, self.tcp_euler is not None))
            out.append((k, ce["robustness"], CALVIN_ACTION_SCALE * (m.T @ normal)))
        return out

    def _cap_rows(self, original, chunk):
        delta = chunk[:, :3] - original[:, :3]
        norms = np.linalg.norm(delta, axis=1) * CALVIN_ACTION_SCALE
        chunk[:, :3] = original[:, :3] + delta * np.minimum(1.0, self.repair_row_cap / np.maximum(norms, 1e-12))[:, None]
        return chunk

    def _repair_linearized(self, joint_angles, task_chunk, tube):
        """Minimum-norm correction of the linearized violated constraints,
        re-linearized until the true tube certifies (at most
        `max_linearized_iters` rounds). "polyak": project onto the worst one at
        a time (closed form); "ce_qp": one small QP over every violated step.
        `tube` is the candidate's tube from certification."""
        original, chunk = task_chunk.copy(), task_chunk.copy()
        rows = min(len(chunk), self.horizon_steps or len(chunk))
        target = self.epsilon + self.repair_margin
        ok = False
        for _ in range(self.max_linearized_iters):
            if self.repair_budget is not None and self._rounds >= self.repair_budget:
                break
            violations = self._violations(joint_angles, chunk, tube, worst_only=self.repair_mode == "polyak")
            if not violations:
                break
            self._rounds += 1
            if self.repair_mode == "polyak":
                k, rho, g = min(violations, key=lambda v: v[1])
                free = k - self.fixed_prefix
                if free <= 0:  # the worst step lies in the committed rows
                    break
                chunk[self.fixed_prefix:k, :3] += (target - rho) / (free * float(g @ g) + 1e-12) * g
            else:
                d = cp.Variable((rows, 3))
                constraints = [rho + cp.sum(d[:k] @ g) >= target for k, rho, g in violations]
                step, _ = _solve_or_none(cp.Problem(cp.Minimize(cp.sum_squares(d)), constraints), d)
                if step is None:
                    return chunk, False
                chunk[:rows, :3] += step
            chunk = self._cap_rows(original, chunk)
            tube, ok = self._certify(joint_angles, chunk)
            if ok:
                return chunk, True
        return chunk, ok

    def _repair(self, joint_angles, task_chunk, tube):
        """(chunk, success) for a candidate whose certification gave `tube`."""
        if self.repair_mode != "gradient":
            return self._repair_linearized(joint_angles, task_chunk, tube)
        original = task_chunk.copy()
        chunk = task_chunk.copy()
        ce = arm_find_counterexample(tube, self.obstacles)
        # step_size/trust_region are meters; chunk columns are raw actions.
        step_size_raw = self.step_size / CALVIN_ACTION_SCALE
        trust_region_raw = self.trust_region / CALVIN_ACTION_SCALE
        for _ in range(self.max_repair_iters):
            k_star = ce["step"]
            direction = self._repair_direction(joint_angles, chunk, ce)
            # Eq. 4: move every row up to the violating step along d.
            chunk[:k_star, :3] = chunk[:k_star, :3] + step_size_raw * direction
            delta = chunk - original
            norm = np.linalg.norm(delta)
            if norm > trust_region_raw:
                chunk = original + delta * (trust_region_raw / norm)

            tube, ok = self._certify(joint_angles, chunk)
            if ok:
                return chunk, True
            ce = arm_find_counterexample(tube, self.obstacles)
        return chunk, False

    def select(self, joint_angles, candidates, scores):
        tubes, mask = zip(*(self._certify(joint_angles, c) for c in candidates))
        certified, mask, repaired = list(mask), list(mask), list(candidates)
        repair_attempted = repair_succeeded = False
        self._rounds = 0
        if not any(mask):
            repair_attempted = True
            order = sorted(range(len(candidates)), key=lambda i: -self._score(joint_angles, candidates[i], scores, i))
            for i in order:
                repaired[i], mask[i] = self._repair(joint_angles, candidates[i], tubes[i])
                if mask[i]:
                    repair_succeeded = True
                    break
        # certified_mask: which candidates passed as proposed (a repaired one was rejected).
        extra = {"repair_attempted": repair_attempted, "repair_succeeded": repair_succeeded,
                 "certified_mask": certified}
        if repair_attempted:  # re-linearizations used, logged in status_counts
            extra["status"] = f"{'ok' if repair_succeeded else 'fail'}_{self._rounds:03d}"
        return self._select_by_mask(joint_angles, repaired, mask, scores, extra_info=extra)


class ArmMPCFilterShield(ArmReachOnlyShield):
    """MPC-Filter, a predictive safety filter (Wabersich & Zeilinger 2021,
    paper ref [33]): the action chunk closest to the policy's top candidate
    (`candidates[0]`, first `horizon_steps` rows) whose predicted arm keeps
    clear of every obstacle and within the joint limits.

    Optimal control problem: minimize ||a - a_policy||^2 subject to, for
    every step k <= H, every arm primitive i and obstacle o,
        dist(primitive_i(q_k(a)), o) >= r_o + r_i + w_bar + model_error
    (constraint tightening by the model-error bound, as in robust MPC), with
    q_k(a) from the shared predictor f-hat (TCP Jacobian). The predictor,
    primitives and tightening are the ones every model-based baseline shares
    (paper: "identical f-hat, epsilon and fallback").

    Solved by SQP: linearize the constraints around the current iterate
    (joint offset at step k = sum of pinv(J_tcp(q_j)) (a_j - a_iter_j); each
    primitive kept SQP_MARGIN beyond a tangent plane of each obstacle),
    solve the QP, re-linearize at its solution; stop at the first iterate
    that satisfies the nonlinear constraints (SQP feasibility check), at
    most `max_sqp_iters` QPs. The policy's chunk is returned unchanged when
    it already satisfies them; no feasible iterate -> total fallback.

    The other candidates are never looked at and are reported admissible."""

    corrects_first_candidate = True  # acts on candidates[0] only (activation/precision use it)

    def __init__(self, obstacles, w_bar, model_error=MODEL_ERROR, horizon_steps=None, max_sqp_iters=5):
        super().__init__(obstacles, w_bar, model_error, horizon_steps)
        self.max_sqp_iters = max_sqp_iters

    def select(self, joint_angles, candidates, scores):
        del scores  # corrects candidates[0], never ranks
        nominal_chunk = np.asarray(candidates[0], dtype=float).copy()
        horizon_chunk = nominal_chunk[:self.horizon_steps]
        corrected_position, iterations, status = self._sqp(joint_angles, horizon_chunk)

        if corrected_position is None:
            mask = [False] + [True] * (len(candidates) - 1)
            return np.zeros_like(nominal_chunk), {
                "fallback": True, "n_admissible": 0, "admissible_mask": mask, "sqp_iterations": iterations,
                "status": status, "repair_attempted": True, "repair_succeeded": False,
            }

        chunk = nominal_chunk.copy()
        chunk[:len(horizon_chunk), :3] = corrected_position
        intervened = iterations > 0
        mask = [not intervened] + [True] * (len(candidates) - 1)
        n_admissible = len(candidates) - (1 if intervened else 0)
        return chunk, {
            "fallback": False, "n_admissible": n_admissible, "admissible_mask": mask, "sqp_iterations": iterations,
            "status": status, "intervened": intervened, "repair_attempted": intervened, "repair_succeeded": intervened,
        }

    def _sqp(self, joint_angles, policy_chunk):
        """(feasible position columns or None, QP iterations used, status);
        feasible = the nonlinear constraints hold (checked with the shared
        predictor). status: unchanged | corrected | qp_<cvxpy status> |
        not_converged."""
        if self._admissible(joint_angles, policy_chunk):
            return policy_chunk[:, :3].copy(), 0, "unchanged"
        iterate = np.asarray(policy_chunk, dtype=float).copy()
        for iteration in range(1, self.max_sqp_iters + 1):
            position, qp_status = self._solve_qp(joint_angles, iterate, policy_chunk[:, :3])
            if position is None:
                return None, iteration, f"qp_{qp_status}"
            iterate = iterate.copy()
            iterate[:, :3] = position
            if self._admissible(joint_angles, iterate):
                return position, iteration, "corrected"
        return None, self.max_sqp_iters, "not_converged"

    def _solve_qp(self, joint_angles, linearization_chunk, target_position):
        """(position columns closest to `target_position` satisfying the
        constraints linearized around `linearization_chunk`, or None; status)."""
        joint_angles = np.asarray(joint_angles, dtype=float)
        linearization_chunk = np.asarray(linearization_chunk, dtype=float)
        horizon = len(linearization_chunk)

        q_lin = nominal_joint_trajectory(joint_angles, linearization_chunk, self.tcp_euler)  # length horizon+1
        geometry = [_arm_geometry(q) for q in q_lin]
        tcp_pinv = [position_joint_map(q_lin[k], self.tcp_euler is not None) for k in range(horizon)]

        a = cp.Variable((horizon, 3))
        step = a - linearization_chunk[:, :3]
        constraints = [a[:self.fixed_prefix] == linearization_chunk[:self.fixed_prefix, :3]] if self.fixed_prefix else []
        dq = 0
        for k in range(1, horizon + 1):
            dq = dq + tcp_pinv[k - 1] @ (step[k - 1] * CALVIN_ACTION_SCALE)
            lo, hi = self._joint_limits(k)
            constraints.append(q_lin[k] + dq >= lo)
            constraints.append(q_lin[k] + dq <= hi)
            for obstacle in self.obstacles:
                for point, jacobian, radius in _iter_primitives_for_obstacle(geometry[k], obstacle):
                    self._add_tangent_constraint(constraints, point, jacobian, dq, obstacle, radius,
                                                 model_error_at(self.model_error, k))

        problem = cp.Problem(cp.Minimize(cp.sum_squares(a - target_position)), constraints)
        return _solve_or_none(problem, a)

    def _add_tangent_constraint(self, constraints, nominal_point, jacobian, dq_k, obstacle, physical_radius,
                                model_error):
        direction = nominal_point - obstacle.center
        norm = np.linalg.norm(direction)
        n = direction / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])
        radius = obstacle.radius + physical_radius + self.w_bar + model_error + self.safety_margin + SQP_MARGIN
        corrected_point = nominal_point + jacobian @ dq_k
        constraints.append(n @ (corrected_point - obstacle.center) >= radius)


class ArmCBFShield(ArmReachOnlyShield):
    """CBF-Shield (Ames et al., paper refs [37]-[39]): a pointwise QP applied
    at each executed control step, minimally correcting the policy's top
    candidate (`candidates[0]`) one action at a time; no lookahead.

    Barrier per reachtube primitive (frames, links, flange->TCP) and
    obstacle: h = distance minus obstacle radius, physical radius, w_bar and
    model_error. Discrete condition, linearized at the step's state:

        n . (J_i pinv(J_tcp) a * CALVIN_ACTION_SCALE) + alpha * h >= 0

    For alpha in (0, 1] and h >= 0 this keeps h(next) >= (1 - alpha) h: small
    alpha intervenes earlier. From inside an obstacle (h < 0) it only slows
    the intrusion. Joint limits are enforced on the same one-step motion.

    Every shield decides once per replan window from the state at decision
    time, so the CBF filters each of the first `steps` rows (the executed
    ones) in turn, at the state the shared predictor gives after the rows
    already filtered. Any infeasible step -> total fallback."""

    corrects_first_candidate = True  # acts on candidates[0] only (activation/precision use it)

    def __init__(self, obstacles, w_bar, model_error=MODEL_ERROR, alpha=1.0, steps=1):
        super().__init__(obstacles, w_bar, model_error)
        self.alpha = alpha
        self.steps = steps

    def select(self, joint_angles, candidates, scores):
        del scores  # corrects candidates[0], never ranks
        nominal_chunk = np.asarray(candidates[0], dtype=float).copy()
        chunk = nominal_chunk.copy()
        q = np.asarray(joint_angles, dtype=float)
        euler = self.tcp_euler
        for row in range(min(self.steps, len(chunk))):
            if row < self.fixed_prefix:
                q, euler = step_joint_config(q, chunk[row], euler)
                continue
            corrected_action, qp_status = self._solve_qp(q, nominal_chunk[row], row + 1, euler)
            if corrected_action is None:
                mask = [False] + [True] * (len(candidates) - 1)
                return np.zeros_like(nominal_chunk), {
                    "fallback": True, "n_admissible": 0, "admissible_mask": mask,
                    "status": f"qp_{qp_status}_row{row}", "repair_attempted": True, "repair_succeeded": False,
                }
            chunk[row, :3] = corrected_action
            q, euler = step_joint_config(q, chunk[row], euler)

        intervened = not np.allclose(chunk[:, :3], nominal_chunk[:, :3], rtol=0.0, atol=QP_INTERVENTION_TOL)
        mask = [not intervened] + [True] * (len(candidates) - 1)
        n_admissible = len(candidates) - (1 if intervened else 0)
        return chunk, {
            "fallback": False, "n_admissible": n_admissible, "admissible_mask": mask,
            "status": "corrected" if intervened else "unchanged",
            "intervened": intervened, "repair_attempted": intervened, "repair_succeeded": intervened,
        }

    def _solve_qp(self, joint_angles, nominal_row, k=1, tcp_euler=None):
        """(corrected (3,) position action at state `joint_angles` (row k-1
        along f-hat, so the next state is k steps ahead; the row's rotation
        columns are kept), or None; status)."""
        joint_angles = np.asarray(joint_angles, dtype=float)
        nominal_row = np.asarray(nominal_row, dtype=float)
        nominal_action = nominal_row[:3]

        geometry = _arm_geometry(joint_angles)
        if tcp_euler is None:
            position_map, dq_rot = np.linalg.pinv(geometry[3]), np.zeros(len(joint_angles))
        else:
            position_map, dq_rot, _ = pose_step_parts(joint_angles, tcp_euler, nominal_row[3:6])

        a = cp.Variable(3)
        # Full one-step joint motion of the candidate action itself (not an
        # offset from the nominal one), so the constraints see the real next state.
        dq = position_map @ (a * CALVIN_ACTION_SCALE) + dq_rot

        lo, hi = self._joint_limits(k)
        constraints = [joint_angles + dq >= lo, joint_angles + dq <= hi]
        for obstacle in self.obstacles:
            for point, jacobian, radius in _iter_primitives_for_obstacle(geometry, obstacle):
                self._add_cbf_constraint(constraints, point, jacobian, dq, obstacle, radius,
                                         model_error_at(self.model_error, k))

        problem = cp.Problem(cp.Minimize(cp.sum_squares(a - nominal_action)), constraints)
        return _solve_or_none(problem, a)

    def _add_cbf_constraint(self, constraints, current_point, jacobian, dq, obstacle, physical_radius, model_error):
        direction = current_point - obstacle.center
        norm = np.linalg.norm(direction)
        n = direction / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])
        margin = obstacle.radius + physical_radius + self.w_bar + model_error + self.safety_margin
        h_current = float(norm - margin)
        dp = jacobian @ dq
        constraints.append(n @ dp + self.alpha * h_current >= 0)


class ArmConfThreshShield(ShieldBase):
    """Conf-Thresh (paper Sec. V.D): reject candidates whose predicted TCP
    after `horizon_steps` rows is farther than `disagreement_threshold` (m)
    from the K candidates' centroid. Uses no obstacle knowledge and gives no
    guarantee; Select picks the best g(a) among the survivors."""

    def __init__(self, disagreement_threshold, horizon_steps):
        self.disagreement_threshold = disagreement_threshold
        self.horizon_steps = horizon_steps
        self.tcp_euler = None  # set by the harness, as for the model-based shields

    def _endpoint(self, joint_angles, task_chunk):
        tube = propagate_arm_tube(joint_angles, task_chunk[:self.horizon_steps], w_bar=0.0, model_error=0.0,
                                  tcp_euler=self.tcp_euler)
        return tube[-1]["fingertip"].b  # predicted TCP

    def select(self, joint_angles, candidates, scores):
        endpoints = [self._endpoint(joint_angles, c) for c in candidates]
        centroid = np.mean(endpoints, axis=0)
        disagreement = [float(np.linalg.norm(e - centroid)) for e in endpoints]
        mask = [d <= self.disagreement_threshold for d in disagreement]
        return self._select_by_mask(joint_angles, candidates, mask, scores, extra_info={"disagreement": disagreement})
