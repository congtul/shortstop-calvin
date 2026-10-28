"""Diagnostic (not tuning): is ShortStop's low one-decision solve rate a
property of the repair heuristic or of its pivot parameters? Same cases as
one_step.py, ShortStop only, trust region x repair iterations."""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import one_step as os_
from shortstop.calvin.config import HORIZON_STEPS, MODEL_ERROR, REPLAN_STEPS
from shortstop.calvin.reach import CALVIN_ACTION_SCALE, nominal_joint_trajectory
from shortstop.calvin.robot_geometry import gripper_tip_position
from shortstop.calvin.shield import ArmRepairShield
from shortstop.calvin import scene_manifest as sm

VARIANTS = [(tr, it) for tr in (0.03, 0.05, 0.10) for it in (1, 3)]

def run(q, kind, seed):
    rng = np.random.default_rng(seed)
    q = np.asarray(q)
    d = rng.normal(size=3); d[2] *= 0.3; d /= np.linalg.norm(d)
    base = np.zeros((os_.CHUNK, 7)); base[:, :3] = d * os_.SPEED / CALVIN_ACTION_SCALE
    tip = gripper_tip_position(q)
    obstacles = os_.layout(kind, tip + 0.14 * d, d, rng)
    from shortstop.calvin.reach import arm_robustness_to_go, propagate_arm_tube
    if not (arm_robustness_to_go(propagate_arm_tube(q, np.zeros((1, 7)), 0.0, 0.0), obstacles) > 0.02
            and not os_.certified(q, base, obstacles, HORIZON_STEPS)):
        return []
    cands = [base + np.concatenate([rng.normal(0, os_.NOISE, (os_.CHUNK, 3)), np.zeros((os_.CHUNK, 4))], axis=1) for _ in range(os_.K)]
    scores = [0.0] * os_.K
    out = []
    for tr, it in VARIANTS:
        sh = ArmRepairShield(obstacles=obstacles, w_bar=0.0, model_error=MODEL_ERROR, horizon_steps=HORIZON_STEPS,
                             epsilon=0.0, trust_region=tr, step_size=0.05, max_repair_iters=it)
        t0 = time.perf_counter(); chunk, info = sh.select(q, cands, scores); ms = (time.perf_counter() - t0) * 1000
        fb = bool(info["fallback"])
        prog = 0.0 if fb else float((gripper_tip_position(nominal_joint_trajectory(q, np.asarray(chunk)[:REPLAN_STEPS])[-1]) - tip) @ d)
        out.append({"kind": kind, "trust_region": tr, "iters": it, "solved": not fb, "progress_m": prog, "ms": ms})
    return out

configs = [sm.START_JOINT_ANGLES.tolist()]
for split in ("val", "test"):
    for s in json.load(open(os_.REPO / f"shortstop/calvin/data/scenes/{split}_scenes.json"))["scenes"]:
        configs.append(s["provenance"]["planner_contact_config"])
kinds = ["sphere", "bar", "box", "corner", "clutter3"]
rows = []
t0 = time.time()
for i, q in enumerate(configs):
    for j, kind in enumerate(kinds):
        rows += run(q, kind, 1000 * i + j)
    if i % 20 == 0:
        print(f"{i}/{len(configs)} configs, {time.time() - t0:.0f}s", flush=True)
with open(Path(__file__).with_name("repair_sensitivity.jsonl"), "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print("REPAIR_DONE", len(rows))
