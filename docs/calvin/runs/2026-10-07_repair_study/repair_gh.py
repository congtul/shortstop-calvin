"""Repair variants G and H (same cases as repair_variants.py).

Both linearize, for every violating step k, the worst capsule-obstacle
distance there: rho_k(a + D) ~ rho_k + sum_{j<=k} g_k . D_j, with
g_k = s * M^T n (n: witness normal, M = J_witness pinv(J_tcp), s = action scale),
the same for every row j <= k (a row's TCP move reaches step k unchanged).

G  Polyak / cyclic projection: project onto the worst violated linearized
   constraint (minimum-norm step D = (target - rho_k) g / |g|^2 on rows <= k),
   re-linearize, repeat (<= 20) until certified.
H  Counterexample QP: minimum-norm D satisfying ALL violated linearized
   constraints at once (one per violating step), re-linearize, repeat (<= 5).
Target = epsilon + 3 mm. No trust region (a 10 cm per-row cap only guards
against runaway).
"""
import sys
import time
from pathlib import Path

import cvxpy as cp
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import repair_variants as rv  # noqa: E402
from shortstop.calvin.config import HORIZON_STEPS, MODEL_ERROR  # noqa: E402
from shortstop.calvin.reach import CALVIN_ACTION_SCALE, arm_find_counterexample, nominal_joint_trajectory  # noqa: E402
from shortstop.calvin.robot_geometry import gripper_tip_jacobian  # noqa: E402
from shortstop.calvin.shield import ArmRepairShield  # noqa: E402

MARGIN = 0.003
ROW_CAP = 0.10


class _LinearizedRepair(ArmRepairShield):
    def __init__(self, obstacles, max_iters):
        super().__init__(obstacles, w_bar=0.0, model_error=MODEL_ERROR, horizon_steps=HORIZON_STEPS, epsilon=0.0,
                         trust_region=1e9, step_size=0.0)
        self.max_iters = max_iters

    def _violations(self, joint_angles, chunk, tube):
        """[(k, rho_k, g_k)] for every step whose worst distance is below target."""
        q_traj = nominal_joint_trajectory(joint_angles, chunk[:self.horizon_steps])
        out = []
        for k in range(1, len(tube)):
            ce = arm_find_counterexample([tube[0], tube[k]], self.obstacles)
            if ce is None or ce["robustness"] >= self.epsilon + MARGIN:
                continue
            normal = ce["witness"] - ce["obstacle"].center
            norm = np.linalg.norm(normal)
            normal = normal / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])  # witness at the center: push up
            q_k = q_traj[k]
            m = self._witness_jacobian(ce["frame"], q_k, ce["obstacle"].center) @ np.linalg.pinv(gripper_tip_jacobian(q_k))
            out.append((k, ce["robustness"], CALVIN_ACTION_SCALE * (m.T @ normal)))
        return out

    def _cap(self, original, chunk):
        delta = chunk[:, :3] - original[:, :3]
        norms = np.linalg.norm(delta, axis=1) * CALVIN_ACTION_SCALE
        chunk[:, :3] = original[:, :3] + delta * np.minimum(1.0, ROW_CAP / np.maximum(norms, 1e-12))[:, None]
        return chunk


class PolyakRepairShield(_LinearizedRepair):
    def __init__(self, obstacles):
        super().__init__(obstacles, max_iters=20)

    def _repair(self, joint_angles, task_chunk, counterexample):
        original, chunk = task_chunk.copy(), task_chunk.copy()
        tube, ok = self._certify(joint_angles, chunk)
        for _ in range(self.max_iters):
            violations = self._violations(joint_angles, chunk, tube)
            if not violations:
                break
            k, rho, g = min(violations, key=lambda v: v[1])
            gain = (self.epsilon + MARGIN - rho) / (k * float(g @ g) + 1e-12)
            chunk[:k, :3] += gain * g
            chunk = self._cap(original, chunk)
            tube, ok = self._certify(joint_angles, chunk)
            if ok:
                return chunk, True
        return chunk, ok


class CounterexampleQPRepairShield(_LinearizedRepair):
    def __init__(self, obstacles):
        super().__init__(obstacles, max_iters=5)

    def _repair(self, joint_angles, task_chunk, counterexample):
        original, chunk = task_chunk.copy(), task_chunk.copy()
        rows = min(len(chunk), self.horizon_steps)
        tube, ok = self._certify(joint_angles, chunk)
        for _ in range(self.max_iters):
            violations = self._violations(joint_angles, chunk, tube)
            if not violations:
                break
            d = cp.Variable((rows, 3))
            cons = [rho + cp.sum(d[:k] @ g) >= self.epsilon + MARGIN for k, rho, g in violations]
            problem = cp.Problem(cp.Minimize(cp.sum_squares(d)), cons)
            try:
                problem.solve()
            except cp.error.SolverError:
                return chunk, False
            if d.value is None:
                return chunk, False
            chunk[:rows, :3] += d.value
            chunk = self._cap(original, chunk)
            tube, ok = self._certify(joint_angles, chunk)
            if ok:
                return chunk, True
        return chunk, ok


def variants(obstacles):
    out = rv.variants(obstacles)
    keep = {k: out[k] for k in ("A_current", "F_all_violations", "mpc_filter", "cbf")}
    keep["G_polyak"] = PolyakRepairShield(obstacles)
    keep["H_ce_qp"] = CounterexampleQPRepairShield(obstacles)
    return keep


if __name__ == "__main__":
    rv.variants = variants
    rv.HERE = HERE
    import json
    import shortstop.calvin.scene_manifest as sm
    import one_step as os_
    configs = [sm.START_JOINT_ANGLES.tolist()]
    for split in ("val", "test"):
        for s in json.load(open(os_.REPO / f"shortstop/calvin/data/scenes/{split}_scenes.json"))["scenes"]:
            configs.append(s["provenance"]["planner_contact_config"])
    rows, t0 = [], time.time()
    for i, q in enumerate(configs):
        for j, kind in enumerate(["sphere", "bar", "gap", "local_cluster"]):
            rows += rv.case(q, kind, 1000 * i + j)
        if i % 20 == 0:
            print(f"{i}/{len(configs)} configs, {time.time() - t0:.0f}s", flush=True)
    with open(HERE / "repair_gh.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("GH_DONE", len(rows), flush=True)
