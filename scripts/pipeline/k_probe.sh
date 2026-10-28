#!/bin/bash
# K probe on the 40 val scenes (lab_notebook.md, "K probe in the final
# configuration"): ShortStop with K = 1, 4, 8 candidates, frozen config
# otherwise. Validation only; K = 4 stays the K of every reported run.
source "$(dirname "$0")/common.sh"
for K in 1 4 8; do
  echo "== K=$K $(date +%H:%M)"
  T +split=probe_local_cluster +baseline=shortstop +n_candidates=$K +tag=kprobe_K$K
done
python scripts/analyze_calvin_k_probe.py
echo K_PROBE_DONE
