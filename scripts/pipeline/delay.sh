#!/bin/bash
# Inference-delay stress test (paper Table 3): first 200 final scenes,
# d = 1..3; unshielded, then ShortStop / MPC-Filter / CBF naive (H = 4) and
# delay-aware (H = 4 + d).
source "$(dirname "$0")/common.sh"
export SHORTSTOP_HORIZON=4
T() { python scripts/run_calvin_test.py +seeds=1 +precision=false +split=final +limit=200 "$@" 2>&1 | grep -E "Traceback|Error" | head -3 || true; }
for d in 1 2 3; do
  echo "== d=$d $(date +%H:%M)"
  T +baseline=unshielded +delay=$d +tag=delay$d
  for mode in naive aware; do
    A=""; [ $mode = aware ] && A="+delay_aware=true"
    T +baseline=shortstop +delay=$d $A +tag=delay${d}_$mode
    T +baseline=mpc_filter +delay=$d $A +tag=delay${d}_$mode
    T +baseline=cbf +delay=$d $A +tag=delay${d}_$mode
  done
done
echo ALL_DONE
