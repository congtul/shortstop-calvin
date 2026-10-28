# Setting up CALVIN + MDT

Tested on Windows 11 + WSL2 Ubuntu 22.04 (kernel 6.6), Intel i5-12450HX,
8 GB RAM visible to WSL, NVIDIA RTX 3050 Laptop 6 GB (driver 577.05).
Native Ubuntu works the same way; the only WSL-specific line is the
`/usr/lib/wsl/lib` entry in `LD_LIBRARY_PATH` (harmless elsewhere).
Disk: about 6 GB for the checkpoint, dataset and encoder cache.

One conda environment (`mdt_env`) runs everything: MDT, the CALVIN simulator
(`calvin_env`, vendored by `mdt_policy`) and ShortStop.

## 1-2. Code and the `mdt_policy` patch

`mdt_policy` is a submodule pinned to `intuitive-robots/mdt_policy@e7a78e7`;
it brings `calvin_env@797142c` and `tacto@dd53360` as its own submodules.
MDT declares `calvin_env` with an SSH URL, so `git clone
--recurse-submodules` fails without a GitHub SSH key. Fetch MDT first, patch
it (the patch switches that URL to HTTPS), then its submodules:

```bash
git clone https://github.com/congtul/shortstop-calvin.git
cd shortstop-calvin
git submodule update --init mdt_policy
bash scripts/apply_mdt_patch.sh
git -C mdt_policy submodule sync --recursive
git -C mdt_policy submodule update --init --recursive
git submodule status --recursive       # e7a78e7, 797142c, dd53360
```

From an archive of this repository (no `.git`), replace the
`git submodule update --init mdt_policy` line with

```bash
rmdir mdt_policy
git clone https://github.com/intuitive-robots/mdt_policy.git mdt_policy
git -C mdt_policy checkout e7a78e744751724d3faa5d113fe8cb5f678c052a
```

`patches/mdt_policy_shortstop.patch` changes no model or evaluation logic:

- `conf/mdt_evaluate.yaml`: checkpoint, dataset and encoder-cache paths
  relative to the repo (`${oc.env:SHORTSTOP_ROOT}`, set when `shortstop.calvin`
  is imported), W&B logging off.
- `mdt/evaluation/mdt_evaluate.py`: MDT's own evaluation ran only when W&B
  logging was on; the GUI call `join_vis_lang` is disabled for headless
  machines. (ShortStop does not use this script; the fix only matters if
  you run MDT's standard evaluation.)
- `.gitmodules`: SSH submodule URL to HTTPS.

## 3. Environment

```bash
bash scripts/setup_calvin_env.sh          # creates mdt_env (Python 3.8)
conda deactivate && conda activate mdt_env
```

The script installs `tacto` and `calvin_env` in editable mode, builds
`pyhash`, installs `mdt_policy/requirements.txt`, then pins
`torchvision==0.15.2`, `networkx==2.8.8` and the ShortStop layer
(`requirements.txt`: NumPy, SciPy, cvxpy and its QP solvers at the versions
of the reported runs). Every install is constrained to the full environment
of those runs, `env/mdt_env.freeze.txt`, so the setup does not pick up newer
releases.

Problems met during setup and their fixes (already in the script):

1. `pyhash`: `use_2to3 is invalid` → install `setuptools==57.5.0` *before*
   building `pyhash-0.9.3`.
2. `networkx==2.2` (pulled in by `gym`/`urdfpy`) uses `np.int`, removed from
   NumPy → pin `networkx==2.8.8`.
3. `libcudnn_cnn_infer.so.8` / `libcuda.so` not found on WSL2 → the script
   writes an `activate.d` hook that adds `/usr/lib/wsl/lib` and the pip
   cuDNN directory to `LD_LIBRARY_PATH` for this environment only.
4. `cv2.imshow` crashes on a headless machine → disabled by the patch.
5. `MulticoreTSNE` / `calvin_models`: not needed; nothing in MDT's or
   ShortStop's code path imports `calvin_models`, so the separate
   `mees/calvin` repository is not required.

## 4. MDT checkpoint

Download the folder `mdtv_1_abcd` (checkpoint `mdtv-1-abcd`, ABCD→D split,
seed 142) from the MDT authors' Google Drive linked in the
[mdt_policy README](https://github.com/intuitive-robots/mdt_policy), and
place it so that this file exists:

```
mdt_policy/pretrained_models/CALVIN_ABCD/mdtv_1_abcd/
├── .hydra/config.yaml
└── saved_models/avg_seq_len=4.64.ckpt
```

## 5. CALVIN debug dataset

```bash
cd mdt_policy/dataset && sh download_data.sh debug && cd ../..
```

ShortStop reads only the environment configuration (scene D), the
normalization statistics and the language embeddings from it; the 166 GB
training sets are not needed.

## 6. Check

```bash
python scripts/smoke_test_calvin.py   # versions, CUDA, imports
python scripts/check_assets.py        # files in place, SHA-256 as in the runs
pytest tests -q                       # 175 unit tests, no simulator needed
```

The Voltron encoder (`v-cond`, ~630 MB) and the DistilBERT tokenizer are
downloaded automatically into `mdt_policy/vcache/` the first time the model
loads; rerun `check_assets.py` afterwards to verify the encoder.

## How ShortStop calls MDT

`MDTVAgent.step(obs, goal)` samples a chunk of 10 actions, caches it and
returns one action per call, so the chunk is never exposed. ShortStop calls
the chunk sampler directly (`denoise_actions`, K candidates in one batch;
`shortstop/calvin/mdt_policy_client.py`) and runs its own loop that executes
`REPLAN_STEPS` rows and proposes again (`shortstop/calvin/experiment.py`).
