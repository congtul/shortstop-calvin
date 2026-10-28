#!/bin/bash
# Final stage: the five shielded methods on the 1000 final scenes, then the
# passive soundness log on the first 300 (paper Table 1, Fig. 2, Sect. 6.2).
source "$(dirname "$0")/common.sh"
export SHORTSTOP_HORIZON=$(cat $P/chosen_H.txt 2>/dev/null || echo 4)
echo "== final (H=$SHORTSTOP_HORIZON) $(date +%H:%M)"
T +split=final +baseline=unshielded
T +split=final +baseline=shortstop
T +split=final +baseline=mpc_filter
T +split=final +baseline=cbf
T +split=final +baseline=conf_thresh
T +split=final +baseline=stl_monitor
echo "== soundness $(date +%H:%M)"
T +split=final +baseline=shortstop +soundness=true +limit=300 +tag=soundness
echo ALL_DONE
