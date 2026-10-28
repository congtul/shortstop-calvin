"""Gate before the long runs: on 12 val scenes the reached arm must stay inside
the certified tube (<= 1 % of rows) and the certified robustness must be a
lower bound (<= 1 % violations of it)."""
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from shortstop.calvin.config import MODEL_ERROR  # noqa: E402
from shortstop.calvin.reach import model_error_at  # noqa: E402

path = REPO_ROOT / "outputs" / "calvin_probe_local_cluster" / "shortstop_hold_gate_smoke.jsonl"
rows = [json.loads(line) for line in path.read_text().splitlines()]
S = np.array([x for r in rows for x in r["soundness"]])
k, ph, pt, dev = S[:, 1].astype(int), S[:, 2], S[:, 3], S[:, 4]
eps = np.array([model_error_at(MODEL_ERROR, kk) for kk in k])
outside, unsound = float(np.mean(dev > eps)), float(np.mean(pt < ph - 1e-9))
print(f"gate: {len(S)} rows, outside tube {outside:.4f}, unsound {unsound:.4f}, "
      f"succ {np.mean([r['reached'] for r in rows]):.2f} viol {np.mean([r['violated'] for r in rows]):.2f}")
for kk in sorted(set(k)):
    m = k == kk
    print(f"  k={kk}: dev p99 {1e3 * np.percentile(dev[m], 99):.1f} mm vs eps {1e3 * model_error_at(MODEL_ERROR, kk):.1f} mm")
sys.exit(0 if outside <= 0.01 and unsound <= 0.01 else 1)
