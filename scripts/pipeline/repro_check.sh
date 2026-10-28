#!/bin/bash
# Level-1 check (REPRODUCE.md): rerun the first 3 final scenes with every
# method and compare each rollout with the published one (~10 min on a laptop GPU).
source "$(dirname "$0")/common.sh"
for b in unshielded shortstop mpc_filter cbf conf_thresh stl_monitor; do
  echo "== $b $(date +%H:%M)"
  T +split=final +limit=3 +baseline=$b +tag=repro
done
python scripts/verify_reproduction.py unshielded shortstop_hold mpc_filter_hold cbf_hold conf_thresh_hold stl_monitor_hold
