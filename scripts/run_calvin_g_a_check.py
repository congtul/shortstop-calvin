"""Does g(a) help? Paired check of the current g(a) (TCP goal distance, the
scene builder's two-phase task goals -- shortstop.calvin.task_goals) against
no g(a) at all, on the frozen val scenes
(shortstop/calvin/data/scenes/val_scenes.json).

Same scenes, same per-scene policy seeds, same shield -- only g(a) differs:
- g_a=on:  Select takes the admissible candidate with the best g(a).
- g_a=off: every score is 0.0, so Select takes the first admissible
           candidate (the policy's own order).
Two vehicles with nothing to tune, so the result is about g(a), not about
how well a shield happens to be tuned:
- best_of_n: ArmConfThreshShield with an infinite threshold -- admits every
  candidate, so Select is purely g(a) (no safety filter).
- stl_monitor: ArmSTLMonitorShield at its chosen epsilon (model_error=0 by
  definition) -- g(a) ranking behind a real safety filter.
MPC-Filter/CBF ignore g(a); ShortStop/Conf-Thresh still need tuning.

Checkpointed: every finished rollout is appended to RESULTS_PATH (one JSON
line) and a re-run skips rollouts already there. No videos -- render a
scene afterwards with scripts/render_obstacle_video.py +scene_id=<id>.

Run from WSL2, inside `mdt_env`:

    cd SHORTSTOP
    python scripts/run_calvin_g_a_check.py                  # everything (resumes)
    python scripts/run_calvin_g_a_check.py +limit=2         # timing check: first 2 scenes
    python scripts/run_calvin_g_a_check.py +summary_only=true
    python scripts/run_calvin_g_a_check.py +seeds=3         # more seeds per scene
"""
import json
import sys
import time
from pathlib import Path

import calvin_env  # noqa: F401 -- import side effects only, matches every other CALVIN script
import hydra
import numpy as np
from pytorch_lightning import seed_everything

REPO_ROOT = Path(__file__).resolve().parents[1]
MDT_POLICY_ROOT = REPO_ROOT / "mdt_policy"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(MDT_POLICY_ROOT))

from shortstop.calvin import scene_manifest as sm  # noqa: E402
from shortstop.calvin.baseline_runner import make_logger, setup_env_and_policy  # noqa: E402
from shortstop.calvin.config import HORIZON_STEPS, N_CANDIDATES, REPLAN_STEPS, STL_MONITOR_EPSILON  # noqa: E402
from shortstop.calvin.experiment import run_calvin_shielded_sequence  # noqa: E402
from shortstop.calvin.shield import ArmConfThreshShield, ArmSTLMonitorShield  # noqa: E402

VEHICLES = ("best_of_n", "stl_monitor")
G_A = ("on", "off")
SPLIT = "val"
# Seeds per scene: k = 0..seeds-1 uses policy_seed + 1000 * k (scene policy
# seeds are base + index, index < 1000, so no collisions).
DEFAULT_SEEDS = 1

SCENES_DIR = REPO_ROOT / "shortstop" / "calvin" / "data" / "scenes"
OUT_DIR = REPO_ROOT / "outputs" / "calvin_g_a_check"
RESULTS_PATH = OUT_DIR / "results.jsonl"


def _make_shield(vehicle):
    if vehicle == "best_of_n":
        return ArmConfThreshShield(disagreement_threshold=float("inf"), horizon_steps=HORIZON_STEPS)
    if vehicle == "stl_monitor":
        return ArmSTLMonitorShield(obstacles=[], epsilon=STL_MONITOR_EPSILON, horizon_steps=HORIZON_STEPS)
    raise ValueError(f"unknown vehicle {vehicle!r}")


def _key(row):
    return (row["vehicle"], row["g_a"], row["scene_id"], row["seed_k"])


def _load_done():
    if not RESULTS_PATH.exists():
        return {}
    done = {}
    for line in RESULTS_PATH.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            done[_key(row)] = row
    return done


def _task_family(task_name):
    for prefix in ("push_into", "push", "lift", "rotate", "turn", "open", "close", "move_slider"):
        if task_name.startswith(prefix):
            return {"open": "door", "close": "door", "move_slider": "door", "turn": "light"}.get(prefix, prefix)
    return "other"


def summarize(rows, log):
    """Per vehicle: g(a) on vs off safe-success / violation, overall, per
    task family, and paired (same scene+seed) win/loss counts."""
    by = {}
    for r in rows:
        by.setdefault((r["vehicle"], r["g_a"]), {})[(r["scene_id"], r["seed_k"])] = r
    for vehicle in VEHICLES:
        on, off = by.get((vehicle, "on"), {}), by.get((vehicle, "off"), {})
        if not on and not off:
            continue
        log(f"[summary] vehicle={vehicle}")
        for name, d in (("g(a) on ", on), ("g(a) off", off)):
            if not d:
                continue
            rs = list(d.values())
            log(f"    {name} n={len(rs):3d} safe-success={np.mean([r['reached'] for r in rs]):.3f} "
                f"violation={np.mean([r['violated'] for r in rs]):.3f} "
                f"activation={np.mean([r['activation_rate'] for r in rs]):.3f} "
                f"mean steps={np.mean([r['steps_taken'] for r in rs]):.0f}")
        if on:
            rs = list(on.values())
            with_goal = [r for r in rs if r["g_a_has_goal"]]
            phase2 = [r for r in with_goal if r["g_a_phase2_step"] is not None]
            log(f"    g(a) on: goal resolved {len(with_goal)}/{len(rs)}, reached phase 2 {len(phase2)}/{len(with_goal)} "
                f"(of the successes: {sum(r['reached'] for r in phase2)}/{sum(r['reached'] for r in with_goal)})")
        paired = sorted(set(on) & set(off))
        if paired:
            on_only = sum(on[k]["reached"] and not off[k]["reached"] for k in paired)
            off_only = sum(off[k]["reached"] and not on[k]["reached"] for k in paired)
            log(f"    paired n={len(paired)}: only on succeeds={on_only}, only off succeeds={off_only}")
            for fam in sorted({_task_family(on[k]["task_name"]) for k in paired}):
                ks = [k for k in paired if _task_family(on[k]["task_name"]) == fam]
                log(f"      {fam:9s} n={len(ks):3d} on={np.mean([on[k]['reached'] for k in ks]):.2f} "
                    f"off={np.mean([off[k]['reached'] for k in ks]):.2f}")


@hydra.main(config_path="../mdt_policy/conf", config_name="mdt_evaluate")
def main(cfg):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = make_logger(OUT_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log")
    done = _load_done()
    if cfg.get("summary_only", False):
        summarize(list(done.values()), log)
        return

    limit = cfg.get("limit", None)
    seeds = int(cfg.get("seeds", DEFAULT_SEEDS))
    from mdt.evaluation.utils import get_env_state_for_initial_condition
    _manifest, scenes = sm.load_scenes(SCENES_DIR / f"{SPLIT}_scenes.json", get_env_state_for_initial_condition)
    scenes = scenes[:limit] if limit else scenes

    jobs = [(vehicle, g_a, scene, k)
            for vehicle in VEHICLES for k in range(seeds) for scene in scenes for g_a in G_A]
    pending = [j for j in jobs if (j[0], j[1], j[2]["scene_id"], j[3]) not in done]
    log(f"[g_a_check] {len(jobs)} rollouts ({len(scenes)} {SPLIT} scenes x {seeds} seeds x {len(G_A)} g(a) arms "
        f"x {len(VEHICLES)} vehicles); {len(pending)} to run")
    if not pending:
        summarize(list(done.values()), log)
        return

    seed_everything(0, workers=True)
    env, policy, task_oracle, val_annotations, lang_embeddings = setup_env_and_policy(cfg, N_CANDIDATES)
    shields = {vehicle: _make_shield(vehicle) for vehicle in VEHICLES}
    started = time.perf_counter()
    with open(RESULTS_PATH, "a", encoding="utf-8") as out:
        for n_done, (vehicle, g_a, scene, k) in enumerate(pending, 1):
            shield = shields[vehicle]
            shield.obstacles = []  # the harness sets the scene's obstacle itself
            seed_everything(scene["policy_seed"] + 1000 * k, workers=True)
            t0 = time.perf_counter()
            attempts = run_calvin_shielded_sequence(
                env, policy, task_oracle, lang_embeddings, scene["initial_state"], scene["eval_sequence"],
                val_annotations, get_env_state_for_initial_condition, shield, ep_len=cfg.ep_len,
                replan_steps=REPLAN_STEPS, horizon_steps=HORIZON_STEPS,
                obstacles=scene["obstacles"],
                measure_intervention_precision=False, use_progress=(g_a == "on"),
            )
            a = attempts[0]
            row = {
                "vehicle": vehicle, "g_a": g_a, "scene_id": scene["scene_id"],
                "seed_k": k, "policy_seed": scene["policy_seed"] + 1000 * k, "task_name": scene["task_name"],
                "violated": bool(a["violated"]), "reached": bool(a["reached"]), "steps_taken": int(a["steps_taken"]),
                "activation_rate": a["n_activated"] / max(a["n_decisions"], 1),
                "g_a_has_goal": bool(a["g_a_has_goal"]), "g_a_phase2_step": a["g_a_phase2_step"],
                "progress_ms_mean": float(np.mean(a["progress_latencies_ms"])) if a["progress_latencies_ms"] else 0.0,
                "seconds": round(time.perf_counter() - t0, 2),
            }
            out.write(json.dumps(row) + "\n")
            out.flush()
            done[_key(row)] = row
            eta_h = (time.perf_counter() - started) / n_done * (len(pending) - n_done) / 3600
            log(f"[g_a_check] {vehicle:11s} g(a)={g_a:3s} {scene['scene_id']} k={k} "
                f"{scene['task_name']:24s} reached={row['reached']!s:5} violated={row['violated']!s:5} "
                f"steps={row['steps_taken']:3d} phase2@{row['g_a_phase2_step']} {row['seconds']:.1f}s "
                f"[{n_done}/{len(pending)}, ETA {eta_h:.1f} h]")
    summarize(list(done.values()), log)
    log.file.close()


if __name__ == "__main__":
    main()
