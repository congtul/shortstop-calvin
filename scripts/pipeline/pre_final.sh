#!/bin/bash
# Pre-final stage on the 40 val scenes: soundness gate -> H sweep {4,6,8} ->
# tune CBF alpha and the Conf-Thresh threshold at the chosen H. Stop and
# review the output before final.sh.
source "$(dirname "$0")/common.sh"
echo "== gate $(date +%H:%M)"
T +split=probe_local_cluster +baseline=shortstop +soundness=true +limit=12 +tag=gate
T +split=probe_local_cluster +baseline=mpc_filter +limit=2 +tag=gate
T +split=probe_local_cluster +baseline=cbf +limit=2 +tag=gate
[ -s outputs/calvin_probe_local_cluster/mpc_filter_hold_gate_smoke.jsonl ] && \
  [ -s outputs/calvin_probe_local_cluster/cbf_hold_gate_smoke.jsonl ] || { echo GATE_FAILED smoke; exit 1; }
python scripts/pipeline/gate_check.py || { echo GATE_FAILED soundness; exit 1; }
echo "== H sweep $(date +%H:%M)"
for H in 4 6 8; do for b in shortstop mpc_filter cbf; do
  SHORTSTOP_HORIZON=$H T +split=probe_local_cluster +baseline=$b +tag=H$H
done; done
python scripts/pipeline/summarize_h.py
python scripts/pipeline/choose_h.py
H=$(cat $P/chosen_H.txt); export SHORTSTOP_HORIZON=$H
echo "== tuning (H=$H) $(date +%H:%M)"
for b in cbf conf_thresh; do
  python scripts/run_calvin_tuning.py +split=probe_local_cluster +baseline=$b +tag=v2 2>&1 | grep -E "chosen|Traceback|Error" | head -3
done
echo PRE_FINAL_DONE
