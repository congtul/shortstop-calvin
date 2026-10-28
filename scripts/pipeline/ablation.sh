#!/bin/bash
# Component ablation and horizon sweep (paper Table 2, Sect. 7).
# Stage "smoke" (3 scenes per config, stop and review) or "full" (first 300
# final scenes). L0 and L4 (H = 4) are the final rows of the same scenes.
source "$(dirname "$0")/common.sh"
STAGE=${1:?smoke or full}
N=$([ "$STAGE" = smoke ] && echo 3 || echo 300)
S=$([ "$STAGE" = smoke ] && echo _try || echo "")
T() { python scripts/run_calvin_test.py +seeds=1 +precision=false +split=final +limit=$N "$@" 2>&1 | grep -E "Traceback|Error" | head -3 || true; }
export SHORTSTOP_HORIZON=4
echo "== ablation $STAGE $(date +%H:%M)"
T +baseline=reach_only +n_candidates=1 +tag=abl_L1$S
T +baseline=reach_only +progress=false +tag=abl_L2$S
T +baseline=reach_only +tag=abl_L3$S
for H in 6 8; do
  SHORTSTOP_HORIZON=$H T +baseline=shortstop +tag=H$H$S
done
[ "$STAGE" = smoke ] && T +baseline=shortstop +tag=H4$S
echo ALL_DONE
