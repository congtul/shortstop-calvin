#!/bin/bash
# Waits on a running stage: exits when its log says done, its gate fails, the
# stage process is gone, or no rollout has been written for 45 min.
#   bash scripts/pipeline/watchdog.sh final      # watches outputs/pipeline/final.out
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
P=$REPO_ROOT/outputs/pipeline/${1:?stage name, e.g. final}.out
count() { cat "$REPO_ROOT"/outputs/pipeline/*.out "$REPO_ROOT"/outputs/calvin_*/*.jsonl 2>/dev/null | wc -l; }
last=$(count); t=$(date +%s)
while true; do
  sleep 300
  grep -qE "ALL_DONE|PRE_FINAL_DONE" "$P" && { echo "DONE $(date +%H:%M)"; tail -30 "$P"; exit 0; }
  grep -q GATE_FAILED "$P" && { echo "GATE_FAILED $(date +%H:%M)"; tail -30 "$P"; exit 3; }
  pgrep -f "(^|[ /])(pre_final|final|delay|ablation)\.sh( |$)" >/dev/null || { echo "DIED $(date +%H:%M)"; tail -30 "$P"; exit 1; }
  n=$(count)
  if [ "$n" -gt "$last" ]; then last=$n; t=$(date +%s)
  elif [ $(( $(date +%s) - t )) -gt 2700 ]; then echo "STALLED $(date +%H:%M)"; tail -20 "$P"; exit 2; fi
done
