"""H sweep table on the 40 val scenes (tube-based shields)."""
import json
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parents[2] / "outputs" / "calvin_probe_local_cluster"
for H in (4, 6, 8):
    for b in ("shortstop", "mpc_filter", "cbf"):
        r = [json.loads(line) for line in (D / f"{b}_hold_H{H}.jsonl").read_text().splitlines()]
        ms = [m for e in r for m in e["select_ms"][1:]]
        print(f"H={H} {b:10s} n={len(r)} success {np.mean([e['reached'] and not e['violated'] for e in r]):.3f} "
              f"violation {np.mean([e['violated'] for e in r]):.3f} fallback-ep {np.mean([e['n_fallback'] > 0 for e in r]):.2f} "
              f"select ms median {np.median(ms):.0f} p95 {np.percentile(ms, 95):.0f} p99 {np.percentile(ms, 99):.0f} max {max(ms):.0f}")
