# Sourced by the pipeline stages: repo root, conda env, and the run helper.
# Every run appends one JSON line per rollout and skips finished ones, so a
# stage can be restarted after an interruption with the same command.
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SHORTSTOP_CONDA_ENV:-mdt_env}"
P=outputs/pipeline
mkdir -p "$P"
# One rollout per scene; errors are echoed, everything else goes to the run's log.
T() { python scripts/run_calvin_test.py +seeds=1 +precision=false "$@" 2>&1 | grep -E "Traceback|Error" | head -3 || true; }
