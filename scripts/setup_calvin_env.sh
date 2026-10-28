#!/bin/bash
# One-environment setup for the CALVIN + MDT pipeline (docs/CALVIN_SETUP.md,
# step 3). Run after the submodules are in place and patched (steps 1-2).
# Every package is constrained to the version of the reported runs
# (env/mdt_env.freeze.txt). Does not download the checkpoint or the dataset
# (steps 4-5).
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_NAME="${1:-mdt_env}"

if [ ! -d "$REPO_ROOT/mdt_policy/calvin_env" ] || [ -z "$(ls -A "$REPO_ROOT/mdt_policy/calvin_env" 2>/dev/null)" ]; then
    echo "mdt_policy/calvin_env is empty -- run 'git submodule update --init --recursive' first." >&2
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | grep -q "^$ENV_NAME "; then
    conda create -y -n "$ENV_NAME" python=3.8
fi
conda activate "$ENV_NAME"

# pip refuses URL constraints (PyOpenGL's pinned commit comes from tacto);
# networkx is forced to 2.8.8 below, over urdfpy's networkx==2.2.
CONSTRAINTS="$(mktemp)"
grep -v -e ' @ ' -e '^networkx==' "$REPO_ROOT/env/mdt_env.freeze.txt" > "$CONSTRAINTS"
export PIP_CONSTRAINT="$CONSTRAINTS"

cd "$REPO_ROOT/mdt_policy/calvin_env/tacto"
pip install -e .
cd "$REPO_ROOT/mdt_policy/calvin_env"
pip install -e .

pip install setuptools==57.5.0
cd "$REPO_ROOT/mdt_policy/pyhash-0.9.3"
python setup.py build
python setup.py install

cd "$REPO_ROOT/mdt_policy"
pip install -r requirements.txt
pip install torchvision==0.15.2
pip install networkx==2.8.8
pip install -r "$REPO_ROOT/requirements.txt"

mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/cuda_libs.sh" <<EOF
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$CONDA_PREFIX/lib/python3.8/site-packages/nvidia/cudnn/lib:\$LD_LIBRARY_PATH
EOF

if ! git -C "$REPO_ROOT/mdt_policy" apply --reverse --check "$REPO_ROOT/patches/mdt_policy_shortstop.patch" 2>/dev/null; then
    bash "$REPO_ROOT/scripts/apply_mdt_patch.sh"
fi
unset PIP_CONSTRAINT
rm -f "$CONSTRAINTS"

echo ""
echo "Done. Re-activate to pick up LD_LIBRARY_PATH: conda deactivate && conda activate $ENV_NAME"
echo "Next: docs/CALVIN_SETUP.md, steps 4-6 (checkpoint, dataset, smoke test)."
