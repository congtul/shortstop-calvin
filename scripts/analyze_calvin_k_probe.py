"""K probe (docs/calvin/lab_notebook.md, "K probe in the final configuration";
paper Sect. 5): ShortStop with K = 1, 4, 8 candidates on the 40 val scenes.
Safe success, violation, fallback, added latency, MDT sampling time and
paired bootstrap tests vs K = 4 (the K of every reported run).

    python scripts/analyze_calvin_k_probe.py
"""
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
VAL = REPO_ROOT / "outputs" / "calvin_probe_local_cluster"
KS = (1, 4, 8)
N_BOOT = 10000


def _load(k):
    return {e["scene_id"]: e for e in map(json.loads, open(VAL / f"shortstop_hold_kprobe_K{k}.jsonl"))}


def _safe(e):
    return e["reached"] and not e["violated"]


def main():
    runs = {k: _load(k) for k in KS}
    keys = sorted(runs[4])
    assert all(set(r) == set(keys) for r in runs.values()), "K values must share scenes"
    boot = np.random.default_rng(0).integers(0, len(keys), (N_BOOT, len(keys)))
    base = np.array([_safe(runs[4][s]) for s in keys], dtype=float)
    print("| K | Safe success | Violation | Fallback | Added ms median / p95 / p99 / max | MDT sampling median ms | vs K = 4 |")
    print("|---|---|---|---|---|---|---|")
    for k in KS:
        r = [runs[k][s] for s in keys]
        safe = np.array([_safe(e) for e in r], dtype=float)
        fallback = sum(e["n_fallback"] for e in r) / sum(e["n_decisions"] for e in r)
        # warm-up decision excluded, as in analyze_calvin_final.py
        added = [s + p for e in r for s, p in list(zip(e["select_ms"], e["progress_ms"]))[1:]]
        propose = [p for e in r for p in e["propose_ms"][1:]]
        paired = "-"
        if k != 4:
            d = (safe - base)[boot].mean(1)
            p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
            paired = (f"{100 * (safe - base).mean():+.1f} pts, p={p:.2f}, K={k} only {int(((safe == 1) & (base == 0)).sum())}, "
                      f"K=4 only {int(((safe == 0) & (base == 1)).sum())}")
        print(f"| {k} | {100 * safe.mean():.1f} | {100 * np.mean([e['violated'] for e in r]):.1f} | "
              f"{100 * fallback:.1f} | {np.median(added):.0f} / {np.percentile(added, 95):.0f} / "
              f"{np.percentile(added, 99):.0f} / {max(added):.0f} | {np.median(propose):.0f} | {paired} |")


if __name__ == "__main__":
    main()
