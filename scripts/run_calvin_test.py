"""Final evaluation of one baseline on the frozen test scenes.

Runs every test scene with `seeds` policy seeds (scene seed + 1000 k), the
baseline's chosen parameters (`+params_from=<tuning *_chosen.json>`; none for
MPC-Filter / unshielded; default: the frozen config.py values), the freeze fallback (writes <baseline>_freeze.*;
`+fallback=retreat` writes <baseline>.*, the pivot's retreat runs) and intervention-precision
measurement on. The summary reports the paper's metrics, pooled and per seed
(mean +- std), with bootstrap CIs, and conservatism cost against the
unshielded rows of the same scenes/seeds (run `+baseline=unshielded` first).

`+split=final` runs final_scenes.json into outputs/calvin_final/ (default:
the pivot test set, outputs/calvin_test/).

Checkpointed: one JSON line per rollout in outputs/calvin_test/<name>.jsonl;
a re-run skips finished rollouts. `+tag=x` suffixes the result name; `+scene_id=<id>` runs one scene; `+n_candidates=K` overrides K (ablation); `+progress=false` scores every candidate 0 (no g(a), ablation); `+soundness=true` logs tube vs reached arm (passive); `+delay=d` inference delay of d steps (stress test), `+delay_aware=true` the shield certifies the committed rows + the chunk over H = 4 + d; `+precision=false` skips the
intervention-precision rollouts (much faster when a shield stalls).

Run from WSL2, inside `mdt_env`:

    cd SHORTSTOP
    python scripts/run_calvin_test.py +baseline=unshielded +seeds=3
    python scripts/run_calvin_test.py +baseline=stl_monitor +seeds=3 \\
        +params_from=outputs/calvin_tuning/stl_monitor_retreat0.02_chosen.json +retreat_step=0.02
    python scripts/run_calvin_test.py +baseline=mpc_filter +seeds=3 +retreat_step=0.02
    python scripts/run_calvin_test.py +baseline=stl_monitor +summary_only=true
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
    bootstrap, make_logger, make_shield, rollout_metrics, run_scene, setup_env_and_policy, warm_up,
)
from shortstop.calvin.config import (  # noqa: E402
    CBF_ALPHA, CONF_THRESH_THRESHOLD, FALLBACK_RETREAT_STEP, HORIZON_STEPS, N_CANDIDATES, SHORTSTOP_PARAMS,
    STL_MONITOR_EPSILON,
)

# The frozen parameters (config.py) each baseline runs with unless +params_from is given.
FROZEN_PARAMS = {"stl_monitor": {"epsilon": STL_MONITOR_EPSILON}, "cbf": {"alpha": CBF_ALPHA},
                 "conf_thresh": {"threshold": CONF_THRESH_THRESHOLD}, "shortstop": SHORTSTOP_PARAMS}

SCENES_DIR = REPO_ROOT / "shortstop" / "calvin" / "data" / "scenes"
OUT_ROOT = REPO_ROOT / "outputs"


def _load(path):
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {(r["scene_id"], r["seed_k"]): r for r in rows}


def _fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, dict):
        return f"median {value['median']:.1f} / mean {value['mean']:.1f} / p95 {value['p95']:.1f} / max {value['max']:.1f}"
    return f"{value:.3f}"


def summarize(name, rows, log):
    unshielded_path = OUT_DIR / "unshielded.jsonl"
    unshielded = list(_load(unshielded_path).values()) if name != "unshielded" and unshielded_path.exists() else None
    pooled = rollout_metrics(rows, unshielded)
    per_seed = {}
    for k in sorted({r["seed_k"] for r in rows}):
        rs = [r for r in rows if r["seed_k"] == k]
        per_seed[k] = rollout_metrics(rs, None if unshielded is None else [u for u in unshielded if u["seed_k"] == k])
    log(f"[summary] {name}: {pooled['n']} rollouts, seeds {sorted(per_seed)}")
    for key in ("safe_success", "violation", "timeout", "activation", "fallback", "intervention_precision",
                "recovery", "conservatism_cost", "mean_steps", "added_ms", "select_ms", "progress_ms", "propose_ms"):
        if key not in pooled:
            continue
        line = f"  {key:22s} pooled {_fmt(pooled[key])}"
        values = [m[key] for m in per_seed.values() if isinstance(m.get(key), (int, float))]
        if len(values) > 1:
            line += f"   per-seed {np.mean(values):.3f} +- {np.std(values):.3f}"
        log(line)
    if pooled.get("status_counts"):
        log(f"  shield status per decision: {pooled['status_counts']}")
    for field, label in (("reached", "safe_success"), ("violated", "violation")):
        mean, lo, hi = bootstrap(rows, field)
        log(f"  bootstrap 95% CI {label}: {mean:.3f} [{lo:.3f}, {hi:.3f}]")
    summary = {"name": name, "pooled": pooled, "per_seed": per_seed}
    (OUT_DIR / f"{name}_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


@hydra.main(config_path="../mdt_policy/conf", config_name="mdt_evaluate")
def main(cfg):
    baseline = cfg.get("baseline", None)
    seeds = int(cfg.get("seeds", 3))
    retreat_step = float(cfg.get("retreat_step", FALLBACK_RETREAT_STEP))
    params_from = cfg.get("params_from", None)
    if params_from is not None:
        params = json.loads((REPO_ROOT / params_from).read_text())["params"]
    else:
        params = dict(FROZEN_PARAMS.get(baseline, {}))
    global OUT_DIR
    split = cfg.get("split", "test")
    OUT_DIR = OUT_ROOT / ("calvin_test" if split == "test" else f"calvin_{split}")
    fallback = cfg.get("fallback", "hold")
    # Unshielded has no fallback, so no suffix.
    name = baseline if fallback == "retreat" or baseline == "unshielded" else f"{baseline}_{fallback}"
    name = name if not cfg.get("tag", None) else f"{name}_{cfg.tag}"
    name = name if not cfg.get("limit", None) else f"{name}_smoke"
    name = name if not cfg.get("scene_id", None) else f"{name}_{cfg.scene_id}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUT_DIR / f"{name}.jsonl"
    log = make_logger(OUT_DIR / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.log")
    done = _load(results_path)
    if cfg.get("summary_only", False):
        summarize(name, list(done.values()), log)
        return

    shield = make_shield(baseline, params)
    from mdt.evaluation.utils import get_env_state_for_initial_condition
    _manifest, scenes = sm.load_scenes(SCENES_DIR / f"{split}_scenes.json", get_env_state_for_initial_condition)
    limit = cfg.get("limit", None)
    scenes = scenes[:limit] if limit else scenes
    scenes = [s for s in scenes if s["scene_id"] == cfg.scene_id] if cfg.get("scene_id", None) else scenes
    jobs = [(scene, k) for k in range(seeds) for scene in scenes]
    pending = [j for j in jobs if (j[0]["scene_id"], j[1]) not in done]
    log(f"[test] {name} params={json.dumps(params)} retreat_step={retreat_step}: {len(scenes)} test scenes x "
        f"{seeds} seeds = {len(jobs)} rollouts, {len(pending)} to run")
    if pending:
        seed_everything(0, workers=True)
        env, policy, task_oracle, val_annotations, lang_embeddings = setup_env_and_policy(
            cfg, 1 if shield is None else int(cfg.get("n_candidates", N_CANDIDATES)))
        warm_up(baseline, params)
        started = time.perf_counter()
        with open(results_path, "a", encoding="utf-8") as out:
            for n_done, (scene, k) in enumerate(pending, 1):
                if shield is not None:
                    shield = make_shield(baseline, params)
                t0 = time.perf_counter()
                row = run_scene(env, policy, task_oracle, val_annotations, lang_embeddings,
                                get_env_state_for_initial_condition, scene, k, shield, cfg.ep_len,
                                fallback=fallback, retreat_step=retreat_step,
                                measure_intervention_precision=bool(cfg.get("precision", True)),
                                log_soundness=bool(cfg.get("soundness", False)),
                                delay_steps=int(cfg.get("delay", 0)),
                                delay_aware=bool(cfg.get("delay_aware", False)),
                                use_progress=bool(cfg.get("progress", True)))
                row.update({"baseline": baseline, "params": params, "fallback": fallback, "retreat_step": retreat_step,
                            "n_candidates": 1 if shield is None else int(cfg.get("n_candidates", N_CANDIDATES)),
                            "horizon_steps": HORIZON_STEPS, "use_progress": bool(cfg.get("progress", True)),
                            "seconds": round(time.perf_counter() - t0, 2)})
                out.write(json.dumps(row) + "\n")
                out.flush()
                done[(row["scene_id"], k)] = row
                eta_h = (time.perf_counter() - started) / n_done * (len(pending) - n_done) / 3600
                log(f"[test] {name} {scene['scene_id']} k={k} {scene['task_name']:24s} reached={row['reached']!s:5} "
                    f"violated={row['violated']!s:5} steps={row['steps_taken']:3d} {row['seconds']:.1f}s "
                    f"[{n_done}/{len(pending)}, ETA {eta_h:.1f} h]")
    summarize(name, list(done.values()), log)
    log.file.close()


if __name__ == "__main__":
    main()
