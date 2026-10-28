"""One-decision comparison of repair variants on sphere layouts (the final
test's obstacle type). Same case construction as one_step.py.

A  current     : 1 step, eta 5 cm per row, chunk-l2 trust region 3 cm
B  iter        : steps that move the path ~2 cm at k* each, re-computed
                 counterexample + gradient, stop at the first certified chunk,
                 at most 12 steps, no trust region
C  iter_row5   : B + per-row cap |delta a_j| <= 5 cm
D  iter_row10  : B + per-row cap |delta a_j| <= 10 cm
plus MPC-Filter and CBF on the same cases.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import one_step as os_  # noqa: E402
from shortstop.calvin import scene_manifest as sm  # noqa: E402
from shortstop.calvin.config import HORIZON_STEPS, MODEL_ERROR, REPLAN_STEPS  # noqa: E402
from shortstop.calvin.reach import (  # noqa: E402
    CALVIN_ACTION_SCALE, arm_find_counterexample, arm_robustness_to_go, nominal_joint_trajectory, propagate_arm_tube,
)
from shortstop.calvin.robot_geometry import gripper_tip_position  # noqa: E402
from shortstop.calvin.shield import ArmCBFShield, ArmMPCFilterShield, ArmRepairShield  # noqa: E402
from shortstop.env import Obstacle  # noqa: E402


class IterativeRepairShield(ArmRepairShield):
    """Repair by small re-linearized steps, stopping at the first certified chunk."""

    def __init__(self, obstacles, path_step=0.02, max_steps=12, row_cap=None):
        super().__init__(obstacles, w_bar=0.0, model_error=MODEL_ERROR, horizon_steps=HORIZON_STEPS, epsilon=0.0,
                         trust_region=1e9, step_size=0.0, max_repair_iters=max_steps)
        self.path_step, self.max_steps, self.row_cap = path_step, max_steps, row_cap

    def _repair(self, joint_angles, task_chunk, counterexample):
        original = task_chunk.copy()
        chunk = task_chunk.copy()
        ce = counterexample
        for _ in range(self.max_steps):
            k = max(1, ce["step"])
            d = self._repair_direction(joint_angles, chunk, ce)
            chunk[:k, :3] += (self.path_step / k) / CALVIN_ACTION_SCALE * d
            if self.row_cap is not None:
                delta = chunk[:, :3] - original[:, :3]
                norms = np.linalg.norm(delta, axis=1) * CALVIN_ACTION_SCALE
                scale = np.minimum(1.0, self.row_cap / np.maximum(norms, 1e-12))
                chunk[:, :3] = original[:, :3] + delta * scale[:, None]
            tube, ok = self._certify(joint_angles, chunk)
            if ok:
                return chunk, True
            ce = arm_find_counterexample(tube, self.obstacles)
        return chunk, False


class LineSearchRepairShield(ArmRepairShield):
    """Fixed direction from the first counterexample; try total path shifts
    2, 4, ... cm at k*, spread evenly over rows 1..k*; first certified wins."""

    def __init__(self, obstacles, shifts=tuple(0.02 * i for i in range(1, 13))):
        super().__init__(obstacles, w_bar=0.0, model_error=MODEL_ERROR, horizon_steps=HORIZON_STEPS, epsilon=0.0,
                         trust_region=1e9, step_size=0.0)
        self.shifts = shifts

    def _repair(self, joint_angles, task_chunk, counterexample):
        k = max(1, counterexample["step"])
        d = self._repair_direction(joint_angles, task_chunk, counterexample)
        for shift in self.shifts:
            chunk = task_chunk.copy()
            chunk[:k, :3] += (shift / k) / CALVIN_ACTION_SCALE * d
            if self._certify(joint_angles, chunk)[1]:
                return chunk, True
        return chunk, False


class AllViolationsRepairShield(IterativeRepairShield):
    """Iterative, direction = depth-weighted sum of the repair directions of
    every violating step (not only the worst), pushing rows up to the latest one."""

    def _repair(self, joint_angles, task_chunk, counterexample):
        chunk = task_chunk.copy()
        tube, ok = self._certify(joint_angles, chunk)
        for _ in range(self.max_steps):
            total, k_max = np.zeros(3), 1
            for k in range(1, len(tube)):
                ce = arm_find_counterexample([tube[0], tube[k]], self.obstacles)
                if ce is None or ce["robustness"] >= self.epsilon:
                    continue
                ce = dict(ce, step=k)
                total += -ce["robustness"] * self._repair_direction(joint_angles, chunk, ce)
                k_max = k
            n = np.linalg.norm(total)
            if n < 1e-12:
                break
            chunk[:k_max, :3] += (self.path_step / k_max) / CALVIN_ACTION_SCALE * total / n
            tube, ok = self._certify(joint_angles, chunk)
            if ok:
                return chunk, True
        return chunk, False


def variants(obstacles):
    return {
        "A_current": ArmRepairShield(obstacles, w_bar=0.0, model_error=MODEL_ERROR, horizon_steps=HORIZON_STEPS,
                                     epsilon=0.0, trust_region=0.03, step_size=0.05),
        "B_iter": IterativeRepairShield(obstacles),
        "C_iter_row5": IterativeRepairShield(obstacles, row_cap=0.05),
        "D_iter_row10": IterativeRepairShield(obstacles, row_cap=0.10),
        "E_linesearch": LineSearchRepairShield(obstacles),
        "F_all_violations": AllViolationsRepairShield(obstacles),
        "mpc_filter": ArmMPCFilterShield(obstacles, w_bar=0.0, model_error=MODEL_ERROR, horizon_steps=HORIZON_STEPS),
        "cbf": ArmCBFShield(obstacles, w_bar=0.0, model_error=MODEL_ERROR, alpha=0.5, steps=REPLAN_STEPS),
    }


def layout(kind, hit, d, rng):
    u, v = os_._basis(d)
    if kind == "sphere":
        return [Obstacle(hit + 0.02 * u, 0.05)]
    if kind == "bar":
        return [Obstacle(hit + (i - 2) * 0.045 * u, 0.03) for i in range(5)]
    if kind == "gap":  # two spheres, surface gap 6-14 cm, centered on the path
        r = rng.uniform(0.04, 0.06, 2)
        g = rng.uniform(0.06, 0.14)
        return [Obstacle(hit + (r[0] + g / 2) * u, r[0]), Obstacle(hit - (r[1] + g / 2) * u, r[1])]
    if kind == "local_cluster":  # 2-3 non-overlapping spheres within 20 cm of the hit point
        for _ in range(100):
            n = int(rng.integers(2, 4))
            unit = lambda w: w / np.linalg.norm(w)  # noqa: E731
            spheres = [Obstacle(hit + rng.uniform(0, 0.2) * unit(rng.normal(size=3)), rng.uniform(0.03, 0.07))
                       for _ in range(n)]
            spheres[0] = Obstacle(hit + 0.02 * u, spheres[0].radius)  # one sphere blocks the path
            ok = all(np.linalg.norm(a.center - b.center) - a.radius - b.radius >= 0.02
                     for i, a in enumerate(spheres) for b in spheres[i + 1:])
            if ok:
                return spheres
        return None
    raise ValueError(kind)


def case(q, kind, seed):
    rng = np.random.default_rng(seed)
    q = np.asarray(q)
    d = rng.normal(size=3)
    d[2] *= 0.3
    d /= np.linalg.norm(d)
    base = np.zeros((os_.CHUNK, 7))
    base[:, :3] = d * os_.SPEED / CALVIN_ACTION_SCALE
    tip = gripper_tip_position(q)
    obstacles = layout(kind, tip + 0.14 * d, d, rng)
    if obstacles is None:
        return []
    clear_now = arm_robustness_to_go(propagate_arm_tube(q, np.zeros((1, 7)), 0.0, 0.0), obstacles) > 0.02
    if not (clear_now and not os_.certified(q, base, obstacles, HORIZON_STEPS)):
        return []
    cands = [base + np.concatenate([rng.normal(0, os_.NOISE, (os_.CHUNK, 3)), np.zeros((os_.CHUNK, 4))], axis=1)
             for _ in range(os_.K)]
    goal = tip + 10 * os_.SPEED * d
    scores = [float(-np.linalg.norm(gripper_tip_position(nominal_joint_trajectory(q, c)[-1]) - goal)) for c in cands]
    out = []
    for name, shield in variants(obstacles).items():
        t0 = time.perf_counter()
        chunk, info = shield.select(q, cands, scores)
        ms = (time.perf_counter() - t0) * 1000
        fallback = bool(info["fallback"])
        rows = REPLAN_STEPS if name == "cbf" else HORIZON_STEPS
        safe = (not fallback) and os_.certified(q, chunk, obstacles, rows)
        executed = nominal_joint_trajectory(q, np.asarray(chunk)[:REPLAN_STEPS])[-1]
        progress = 0.0 if fallback else float((gripper_tip_position(executed) - tip) @ d)
        dev = 0.0 if fallback else float(np.abs(np.asarray(chunk)[:REPLAN_STEPS, :3] - cands[0][:REPLAN_STEPS, :3]).max()
                                         * CALVIN_ACTION_SCALE)
        out.append({"kind": kind, "method": name, "safe": safe, "moving": safe and progress > 0.005,
                    "progress_m": progress, "row_dev_m": dev, "ms": ms})
    return out


def main():
    configs = [sm.START_JOINT_ANGLES.tolist()]
    for split in ("val", "test"):
        for s in json.load(open(os_.REPO / f"shortstop/calvin/data/scenes/{split}_scenes.json"))["scenes"]:
            configs.append(s["provenance"]["planner_contact_config"])
    rows, t0 = [], time.time()
    for i, q in enumerate(configs):
        for j, kind in enumerate(["sphere", "bar", "gap", "local_cluster"]):
            rows += case(q, kind, 1000 * i + j)
        if i % 20 == 0:
            print(f"{i}/{len(configs)} configs, {time.time() - t0:.0f}s", flush=True)
    with open(HERE / "repair_variants.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("VARIANTS_DONE", len(rows), flush=True)


if __name__ == "__main__":
    main()
