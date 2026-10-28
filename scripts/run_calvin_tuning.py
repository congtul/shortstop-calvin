"""Tune one baseline's hyperparameters on the frozen val scenes.

Every config of the baseline's grid (baseline_runner.GRIDS) runs all val
scenes with the scenes' own policy seeds (+1000 k for seed k), g(a) on, the
freeze fallback (`+fallback=retreat` for the retreat ablation) and the
protocol constants of config.py. Each decision also
logs the candidates' TCP disagreement (Conf-Thresh's statistic).

Selection, among the configs with the most seeds run: highest safe success;
configs within SELECTION_TOLERANCE of the best are tied and the lowest shield
activation wins. Written to <name>_summary.json (all configs) and
<name>_chosen.json, where <name> = <baseline>_<fallback><retreat step>.

Checkpointed: one JSON line per rollout in outputs/calvin_tuning/<name>.jsonl;
a re-run skips finished rollouts. No videos (render_obstacle_video.py).

Run from WSL2, inside `mdt_env`:

    cd SHORTSTOP
    python scripts/run_calvin_tuning.py +baseline=stl_monitor
    python scripts/run_calvin_tuning.py +baseline=stl_monitor +limit=2                     # timing check
    python scripts/run_calvin_tuning.py +baseline=stl_monitor +seeds=3 "+configs=[0,1]"    # refine the top configs
    python scripts/run_calvin_tuning.py +baseline=stl_monitor +summary_only=true
    python scripts/run_calvin_tuning.py +baseline=stl_monitor +fallback=freeze             # or +retreat_step=0.02
"""
import json
import sys
import time
from pathlib import Path

import calvin_env  # noqa: F401 -- must be imported before mdt_policy/ goes on sys.path (it contains a calvin_env/ dir)
import hydra
import numpy as np
from pytorch_lightning import seed_everything

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "mdt_policy"))

from shortstop.calvin import scene_manifest as sm  # noqa: E402
from shortstop.calvin.baseline_runner import (  # noqa: E402
    GRIDS, make_logger, make_shield, rollout_metrics, run_scene, setup_env_and_policy, warm_up,
)
from shortstop.calvin.config import FALLBACK_RETREAT_STEP, HORIZON_STEPS, N_CANDIDATES  # noqa: E402
from shortstop.calvin.shield import ArmConfThreshShield  # noqa: E402

SCENES_DIR = REPO_ROOT / "shortstop" / "calvin" / "data" / "scenes"
OUT_DIR = REPO_ROOT / "outputs" / "calvin_tuning"
SELECTION_TOLERANCE = 0.04


def run_name(baseline, fallback, retreat_step):
    return f"{baseline}_retreat{retreat_step:g}" if fallback == "retreat" else f"{baseline}_{fallback}"


class _DisagreementLog:
    """Wraps shield.select to record each candidate's TCP distance to the
    candidates' centroid after HORIZON_STEPS rows."""

    def __init__(self, shield):
        probe = ArmConfThreshShield(disagreement_threshold=float("inf"), horizon_steps=HORIZON_STEPS)
        self.values = []
        original = shield.select

        def select(joint_angles, candidates, scores):
            endpoints = np.array([probe._endpoint(joint_angles, c) for c in candidates])
            self.values.extend(np.linalg.norm(endpoints - endpoints.mean(axis=0), axis=1).tolist())
            return original(joint_angles, candidates, scores)

        shield.select = select


def _key(row):
    return (row["config_index"], row["scene_id"], row["seed_k"])


def load_rows(path):
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {_key(r): r for r in rows}


def summarize(name, rows, log):
    """Per-config metrics + the chosen config; writes <name>_summary/_chosen.json."""
    by_config = {}
    for r in rows:
        by_config.setdefault(r["config_index"], []).append(r)
    stats = []
    for index in sorted(by_config):
        rs = by_config[index]
        m = rollout_metrics(rs)
        stats.append({"config_index": index, "params": rs[0]["params"],
                      "seeds": len({r["seed_k"] for r in rs}), **m})
    log(f"[summary] {name}")
    for s in stats:
        log(f"  #{s['config_index']:2d} {json.dumps(s['params']):58s} n={s['n']:3d} safe-success={s['safe_success']:.3f} "
            f"violation={s['violation']:.3f} timeout={s['timeout']:.3f} activation={s['activation']:.3f} "
            f"fallback={s['fallback']:.3f} steps={s['mean_steps']:.0f} select_ms={(s['select_ms'] or {}).get('mean', float('nan')):.1f}"
            + (f" status={s['status_counts']}" if s.get("status_counts") else ""))
    chosen = None
    if stats:
        most_seeds = max(s["seeds"] for s in stats)
        pool = [s for s in stats if s["seeds"] == most_seeds]
        best = max(s["safe_success"] for s in pool)
        tied = [s for s in pool if s["safe_success"] >= best - SELECTION_TOLERANCE]
        chosen = min(tied, key=lambda s: (s["activation"], -s["safe_success"]))
        log(f"  -> chosen #{chosen['config_index']} {json.dumps(chosen['params'])} among {len(pool)} config(s) "
            f"with {most_seeds} seed(s): best safe-success {best:.3f}, {len(tied)} within {SELECTION_TOLERANCE}")
        (OUT_DIR / f"{name}_chosen.json").write_text(json.dumps(chosen, indent=1))
    (OUT_DIR / f"{name}_summary.json").write_text(json.dumps(stats, indent=1))
    pct = [r["disagreement_percentiles"] for r in rows if r.get("disagreement_percentiles")]
    if pct:
        log("  TCP disagreement per candidate (m, mean of per-rollout percentiles): "
            + " ".join(f"{k}={np.mean([p[k] for p in pct]):.4f}" for k in pct[0]))
    return stats, chosen


@hydra.main(config_path="../mdt_policy/conf", config_name="mdt_evaluate")
def main(cfg):
    baseline = cfg.get("baseline", None)
    if baseline not in GRIDS:
        raise ValueError(f"pass +baseline=<one of {sorted(GRIDS)}>")
    grid = GRIDS[baseline]
    config_indices = list(cfg.get("configs", range(len(grid))))
    seeds = int(cfg.get("seeds", 1))
    limit = cfg.get("limit", None)
    fallback = cfg.get("fallback", "hold")
    retreat_step = float(cfg.get("retreat_step", FALLBACK_RETREAT_STEP))
    name = run_name(baseline, fallback, retreat_step)
    name = name if not cfg.get("tag", None) else f"{name}_{cfg.tag}"
    # +split=<name> tunes on <name>_scenes.json into outputs/calvin_tuning_<name>/ (default: the pivot val).
    split = cfg.get("split", "val")
    global OUT_DIR
    if split != "val":
        OUT_DIR = OUT_DIR.with_name(f"calvin_tuning_{split}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUT_DIR / f"{name}.jsonl"
    log = make_logger(OUT_DIR / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.log")
    done = load_rows(results_path)
    if cfg.get("summary_only", False):
        summarize(name, list(done.values()), log)
        return

    from mdt.evaluation.utils import get_env_state_for_initial_condition
    _manifest, scenes = sm.load_scenes(SCENES_DIR / f"{split}_scenes.json", get_env_state_for_initial_condition)
    scenes = scenes[:limit] if limit else scenes
    jobs = [(i, scene, k) for k in range(seeds) for i in config_indices for scene in scenes]
    pending = [j for j in jobs if (j[0], j[1]["scene_id"], j[2]) not in done]
    log(f"[tuning] {name}: {len(config_indices)} configs x {len(scenes)} val scenes x {seeds} seeds = "
        f"{len(jobs)} rollouts, {len(pending)} to run")
    if pending:
        seed_everything(0, workers=True)
        env, policy, task_oracle, val_annotations, lang_embeddings = setup_env_and_policy(cfg, N_CANDIDATES)
        for index in config_indices:
            warm_up(baseline, grid[index])
        started = time.perf_counter()
        with open(results_path, "a", encoding="utf-8") as out:
            for n_done, (index, scene, k) in enumerate(pending, 1):
                params = grid[index]
                shield = make_shield(baseline, params)
                disagreement = _DisagreementLog(shield)
                t0 = time.perf_counter()
                row = run_scene(env, policy, task_oracle, val_annotations, lang_embeddings,
                                get_env_state_for_initial_condition, scene, k, shield, cfg.ep_len,
                                fallback=fallback, retreat_step=retreat_step)
                values = np.asarray(disagreement.values)
                row.update({
                    "baseline": baseline, "fallback": fallback, "retreat_step": retreat_step,
                    "config_index": index, "params": params, "seconds": round(time.perf_counter() - t0, 2),
                    "disagreement_percentiles": {f"p{p}": float(np.percentile(values, p))
                                                 for p in (50, 75, 90, 95, 99)} if len(values) else {},
                })
                out.write(json.dumps(row) + "\n")
                out.flush()
                done[_key(row)] = row
                eta_h = (time.perf_counter() - started) / n_done * (len(pending) - n_done) / 3600
                log(f"[tuning] {name} #{index} {json.dumps(params)} {scene['scene_id']} k={k} "
                    f"{scene['task_name']:24s} reached={row['reached']!s:5} violated={row['violated']!s:5} "
                    f"steps={row['steps_taken']:3d} act={row['n_activated']}/{row['n_decisions']} "
                    f"{row['seconds']:.1f}s [{n_done}/{len(pending)}, ETA {eta_h:.1f} h]")
    summarize(name, list(done.values()), log)
    log.file.close()


if __name__ == "__main__":
    main()
