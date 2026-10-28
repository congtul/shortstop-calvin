"""Pick H on the 40 val scenes: highest mean task success over the three
tube-based shields (ShortStop, MPC-Filter, CBF); ties -> H = 8. Writes
outputs/pipeline/chosen_H.txt."""
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
D = REPO_ROOT / "outputs" / "calvin_probe_local_cluster"
mean = {}
for H in (4, 6, 8):
    s = [np.mean([json.loads(line)["reached"] for line in (D / f"{b}_hold_H{H}.jsonl").read_text().splitlines()])
         for b in ("shortstop", "mpc_filter", "cbf")]
    mean[H] = float(np.mean(s))
    print(f"H={H}: shortstop/mpc/cbf success {np.round(s, 3).tolist()} mean {mean[H]:.3f}")
best = max(mean.values())
H = 8 if mean[8] >= best - 1e-9 else max(h for h in mean if mean[h] >= best - 1e-9)
print("chosen H", H)
(REPO_ROOT / "outputs" / "pipeline").mkdir(parents=True, exist_ok=True)
(REPO_ROOT / "outputs" / "pipeline" / "chosen_H.txt").write_text(str(H))
