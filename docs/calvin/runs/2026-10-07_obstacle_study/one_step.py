"""One-decision study: does a shield still find a safe, forward-moving chunk
as the obstacle set gets more complex? No simulator: real arm configurations
(reset pose + the RRT contact configs stored in the pivot scenes), a nominal
chunk moving 1.5 cm/row, K=8 candidates with mm spread (like MDT), and an
obstacle layout that the nominal chunk is guaranteed to hit.
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from shortstop.calvin import scene_manifest as sm  # noqa: E402
from shortstop.calvin.config import HORIZON_STEPS, MODEL_ERROR, REPLAN_STEPS  # noqa: E402
from shortstop.calvin.reach import (  # noqa: E402
    CALVIN_ACTION_SCALE, arm_robustness_to_go, nominal_joint_trajectory, propagate_arm_tube,
)
from shortstop.calvin.robot_geometry import gripper_tip_position, within_joint_limits  # noqa: E402
from shortstop.calvin.shield import (  # noqa: E402
    ArmCBFShield, ArmMPCFilterShield, ArmRepairShield, ArmSTLMonitorShield,
)
from shortstop.env import Obstacle  # noqa: E402

K, CHUNK, SPEED, NOISE = 8, 10, 0.015, 0.03


def shields():
    return {
        "stl_monitor": ArmSTLMonitorShield(obstacles=[], epsilon=0.01, horizon_steps=HORIZON_STEPS),
        "shortstop": ArmRepairShield(obstacles=[], w_bar=0.0, model_error=MODEL_ERROR, horizon_steps=HORIZON_STEPS,
                                     epsilon=0.0, trust_region=0.03, step_size=0.05),
        "mpc_filter": ArmMPCFilterShield(obstacles=[], w_bar=0.0, model_error=MODEL_ERROR, horizon_steps=HORIZON_STEPS),
        "cbf": ArmCBFShield(obstacles=[], w_bar=0.0, model_error=MODEL_ERROR, alpha=0.5, steps=REPLAN_STEPS),
    }


def _basis(d):
    u = np.cross(d, [0, 0, 1.0])
    u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-6 else np.array([1.0, 0, 0])
    return u, np.cross(d, u)


def box(center, size, axes, r=0.025):
    """Spheres of radius r covering the surface of a box (size = full extents)."""
    spheres = []
    half = np.asarray(size) / 2
    for face_axis in range(3):
        for sign in (-1, 1):
            others = [a for a in range(3) if a != face_axis]
            n0 = max(2, int(np.ceil(size[others[0]] / (1.6 * r))))
            n1 = max(2, int(np.ceil(size[others[1]] / (1.6 * r))))
            for s0 in np.linspace(-half[others[0]], half[others[0]], n0):
                for s1 in np.linspace(-half[others[1]], half[others[1]], n1):
                    p = center + sign * half[face_axis] * axes[face_axis] + s0 * axes[others[0]] + s1 * axes[others[1]]
                    spheres.append(Obstacle(p, r))
    return spheres


def layout(kind, hit_point, d, rng):
    u, v = _basis(d)
    if kind == "sphere":
        return [Obstacle(hit_point + 0.02 * u, 0.05)]
    if kind == "bar":
        return [Obstacle(hit_point + (i - 2) * 0.045 * u, 0.03) for i in range(5)]
    if kind == "box":
        size = rng.uniform(0.08, 0.15, 3)
        return box(hit_point + 0.5 * size[0] * d, size, [d, u, v])
    if kind == "corner":
        size = np.array([0.12, 0.06, 0.12])
        return box(hit_point + 0.06 * d + 0.08 * u, size, [d, u, v]) + box(hit_point + 0.12 * d - 0.06 * u, size[[1, 0, 2]], [d, u, v])
    if kind == "clutter3":
        out = [Obstacle(hit_point + 0.02 * u, 0.04)]
        out += box(hit_point + 0.10 * d + 0.12 * u, np.array([0.1, 0.08, 0.1]), [d, u, v])
        out += [Obstacle(hit_point - 0.04 * d - 0.12 * v, 0.05)]
        return out
    raise ValueError(kind)


def certified(q, chunk, obstacles, rows):
    tube = propagate_arm_tube(q, np.asarray(chunk)[:rows], w_bar=0.0, model_error=MODEL_ERROR)
    traj = nominal_joint_trajectory(q, np.asarray(chunk)[:rows])
    return arm_robustness_to_go(tube, obstacles) >= 0.0 and all(within_joint_limits(x) for x in traj[1:])


def case(args):
    q, kind, seed = args
    rng = np.random.default_rng(seed)
    q = np.asarray(q)
    d = rng.normal(size=3)
    d[2] *= 0.3
    d /= np.linalg.norm(d)
    base = np.zeros((CHUNK, 7))
    base[:, :3] = d * SPEED / CALVIN_ACTION_SCALE
    tip = gripper_tip_position(q)
    obstacles = layout(kind, tip + 0.14 * d, d, rng)
    clear_now = arm_robustness_to_go(propagate_arm_tube(q, np.zeros((1, 7)), 0.0, 0.0), obstacles) > 0.02
    nominal_hits = not certified(q, base, obstacles, HORIZON_STEPS)
    if not (clear_now and nominal_hits):
        return None
    candidates = [base + np.concatenate([rng.normal(0, NOISE, (CHUNK, 3)), np.zeros((CHUNK, 4))], axis=1) for _ in range(K)]
    scores = [float(-np.linalg.norm(gripper_tip_position(nominal_joint_trajectory(q, c)[-1]) - (tip + 10 * SPEED * d)))
              for c in candidates]
    out = []
    for name, shield in shields().items():
        shield.obstacles = obstacles
        t0 = time.perf_counter()
        chunk, info = shield.select(q, candidates, scores)
        ms = (time.perf_counter() - t0) * 1000
        rows = REPLAN_STEPS if name == "cbf" else HORIZON_STEPS
        fallback = bool(info["fallback"])
        safe = (not fallback) and certified(q, chunk, obstacles, rows)
        executed = nominal_joint_trajectory(q, np.asarray(chunk)[:REPLAN_STEPS])[-1]
        progress = float((gripper_tip_position(executed) - tip) @ d) if not fallback else 0.0
        out.append({"kind": kind, "method": name, "n_spheres": len(obstacles), "fallback": fallback,
                    "safe_chunk": safe, "progress_m": progress, "ms": ms, "status": info.get("status")})
    return out


def main():
    configs = [sm.START_JOINT_ANGLES.tolist()]
    for split in ("val", "test"):
        for s in json.load(open(REPO / f"shortstop/calvin/data/scenes/{split}_scenes.json"))["scenes"]:
            configs.append(s["provenance"]["planner_contact_config"])
    kinds = ["sphere", "bar", "box", "corner", "clutter3"]
    jobs = [(q, kind, 1000 * i + j) for i, q in enumerate(configs) for j, kind in enumerate(kinds)]
    t0 = time.time()
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if workers > 1:
        with Pool(workers) as pool:
            results = [r for rs in pool.map(case, jobs, chunksize=2) if rs for r in rs]
    else:
        results = []
        for n, job in enumerate(jobs, 1):
            rs = case(job)
            results += rs or []
            if n % 50 == 0:
                print(f"{n}/{len(jobs)} cases, {len(results)} rows", flush=True)
    path = Path(__file__).with_name("one_step.jsonl")
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"{len(results)} rows in {time.time() - t0:.0f}s -> {path}")


if __name__ == "__main__":
    main()
