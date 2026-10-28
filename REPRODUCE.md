# Reproducing the CALVIN results

Four levels, from re-deriving every number of the paper in a minute to
rebuilding the benchmark from scratch. Runtimes are for the machine of the
reported runs (Intel i5-12450HX, RTX 3050 Laptop 6 GB, WSL2); the shields run
single-threaded on the CPU, the policy on the GPU, and one simulation job runs
at a time.

| Level | What | Needs | Time |
|---|---|---|---|
| 0 | every table and figure from the released rollouts | Python 3.8+ with NumPy | < 1 min |
| 1 | rerun 3 test scenes with every method, compare rollout by rollout | full setup + GPU | ~10 min |
| 2 | rerun the experiments | full setup + GPU | 3 h to 34 h per stage |
| 3 | rebuild the frozen scenes and the model-error calibration | full setup | scenes 5 min (CPU); rollouts and calibration GPU hours |

## Data

Attached to the release [v1.0](https://github.com/congtul/shortstop-calvin/releases/tag/v1.0) of this repository:

| Archive | Size | Contents |
|---|---|---|
| `shortstop_calvin_results.zip` | 7.9 MB | every rollout behind the paper (final, soundness, delay, ablation, horizon), the validation runs that fixed H, K and the tuned parameters, the candidate-spread log, the ε_k and joint-hold calibration runs, stage logs |
| `shortstop_calvin_reference_rollouts.zip` | 92 MB | obstacle-free unshielded rollouts the scene builder samples from (level 3 only) |
| `shortstop_calvin_superseded_v1.zip` | 3.4 MB | the first final run, superseded after the theory audit; for the record |

Unzip in the repository root; the archives create `outputs/...`. Each one
carries a SHA-256 manifest of its files:

```bash
unzip shortstop_calvin_results.zip
sha256sum -c --quiet outputs/shortstop_calvin_results.sha256
```

`python scripts/pack_artifacts.py` rebuilds the archives from a local
`outputs/`.

## Level 0: numbers from the released rollouts

No simulator, no GPU: only NumPy and the `shortstop` package.

```bash
python scripts/analyze_calvin_final.py   # Table 1, Sects. 6.1-6.2: paired tests, Holm, conservatism, latency,
                                         # collisions, soundness audit, Corollary 2 bound, scene coverage
python scripts/analyze_calvin_extra.py   # Tables 2 (ablation), 3 (delay), horizon sweep (Sect. 7)
python scripts/analyze_calvin_k_probe.py # K probe on the 40 val scenes (Sect. 5)
python docs/paper/shortstop/figures/make_fig2.py   # Fig. 2 and the numbers quoted from it
```

`analyze_calvin_final.py` prints the main table recorded in
[`docs/calvin/runs/2026-10-14_final_v2/README.md`](docs/calvin/runs/2026-10-14_final_v2/README.md)
and every other number of Sects. 5-6 that comes from the main run, the
soundness log or the calibration data. `analyze_calvin_extra.py` and
`make_fig2.py` rewrite `docs/paper/shortstop/tables/*.tex` and
`docs/paper/shortstop/figures/fig2*` (`git diff` shows nothing if they match
the paper). The paper (Springer LNCS) builds with
`latexmk -pdf main.tex` in `docs/paper/shortstop/`; `main.pdf` is that build
(Tectonic also builds it, with Latin Modern fonts and slightly different line breaks).

## Setup (levels 1-3)

[`docs/CALVIN_SETUP.md`](docs/CALVIN_SETUP.md): submodules, the MDT patch,
the `mdt_env` environment, the `mdtv-1-abcd` checkpoint and the CALVIN debug
dataset. Then:

```bash
conda activate mdt_env
python scripts/check_assets.py           # every asset in place, same files as the runs
pytest tests -q
```

## Level 1: rollout-by-rollout check

```bash
bash scripts/pipeline/repro_check.sh
```

Runs the first 3 test scenes with each of the six methods
(`outputs/calvin_final/*_repro_smoke.jsonl`) and compares every outcome field
(success, collision, steps, decisions, fallbacks, shield statuses, minimum
clearance, ...) with the released rollouts; it ends with `REPRODUCED`.
Every rollout is seeded by its scene and the simulator is deterministic, so
on the same software stack the rollouts are identical, not just
statistically close. Different GPU kernels or library versions can change
the policy's samples in the last bits and, through them, individual
rollouts; then compare the aggregates of level 2 instead.

## Level 2: the experiments

Each stage is one script; every run appends one JSON line per rollout to
`outputs/calvin_*/<run>.jsonl` and skips finished rollouts, so an
interrupted stage resumes with the same command. Start a stage in the
background and, optionally, a watchdog that returns when it ends, fails or
stalls:

```bash
mkdir -p outputs/pipeline
nohup bash scripts/pipeline/final.sh > outputs/pipeline/final.out 2>&1 &
bash scripts/pipeline/watchdog.sh final
```

| Stage | Script | Produces | Time |
|---|---|---|---|
| pre-final (val): soundness gate, H sweep {4,6,8}, CBF / Conf-Thresh tuning | `pre_final.sh` | `outputs/calvin_probe_local_cluster/`, `outputs/calvin_tuning_probe_local_cluster/`, `outputs/pipeline/chosen_H.txt` | 3 h |
| K probe (val): ShortStop with K = 1, 4, 8 | `k_probe.sh` | `outputs/calvin_probe_local_cluster/shortstop_hold_kprobe_K<K>.jsonl` | 1 h |
| main comparison, 1000 test scenes, then the soundness log (300) | `final.sh` | `outputs/calvin_final/<method>_hold.jsonl`, `unshielded.jsonl`, `shortstop_hold_soundness_smoke.jsonl` | 34 h + 1.5 h (STL-Monitor alone 16 h) |
| inference delay d = 1..3, 200 scenes | `delay.sh` | `outputs/calvin_final/*_delay<d>_*` | 19 h |
| ablation ladder and H = 6, 8, 300 scenes | `ablation.sh smoke`, then `ablation.sh full` | `outputs/calvin_final/reach_only_hold_abl_*`, `shortstop_hold_H<H>_smoke*` | 23 h |

Then run the analysis scripts of level 0. Notes:

- `final.sh` and `delay.sh` use the frozen parameters in
  `shortstop/calvin/config.py` (H = 4, CBF α = 0.5, Conf-Thresh 5 mm), which
  are what `pre_final.sh` selects; running `pre_final.sh` first re-derives
  them.
- The released unshielded test rows come from the first final run (same
  code path; the unshielded policy has no fallback, the only difference is
  that label). `final.sh` reruns them.
- A single scene: `python scripts/run_calvin_test.py +split=final +seeds=1
  +precision=false +baseline=shortstop +scene_id=final_0010`
  (baselines: `unshielded`, `shortstop`, `mpc_filter`, `cbf`, `conf_thresh`,
  `stl_monitor`, `reach_only`; options `+delay=d`, `+delay_aware=true`,
  `+n_candidates=K`, `+progress=false`, `+soundness=true`, `+limit=N`).
- Video of a scene, for inspection only:
  `python scripts/render_obstacle_video.py +scene_id=final_0010 +baseline=shortstop`.

## Level 3: rebuilding the benchmark

Not needed to check the paper's numbers; the frozen scenes and ε_k are in
the repository. To rebuild them:

1. Static furniture boxes: `python scripts/precompute_static_target_sweeps.py`
   → `shortstop/calvin/data/calvin_task_D_D_target_sweeps.json`.
2. Reference rollouts: `python scripts/collect_calvin_unshielded_rollouts.py`
   → `outputs/unshielded_rollouts/` (or unzip
   `shortstop_calvin_reference_rollouts.zip`).
3. Door contact points: `python scripts/build_door_contact_offsets.py`
   → `shortstop/calvin/data/door_contact_offsets.json`.
4. Scenes: `python scripts/build_calvin_scene_manifest.py probe_local_cluster 40`
   and `python scripts/build_calvin_scene_manifest.py final 1000`
   (CPU, 30 s and 4 min; `final` draws only initial states no other split
   uses, so it reads the other scene files). They overwrite the frozen
   files; `git diff shortstop/calvin/data/scenes` should show only the
   `created` timestamp.
5. Model error: `python scripts/calibrate_arm_model_error.py num_sequences=50`
   (~30 min) → `outputs/calibrate_arm_model_error_runs/run_<time>/calibration.json`;
   `pose.model_error` is `MODEL_ERROR` in `config.py` (the released run is
   `run_20261012_120501` in the results archive).

Rules and parameters of each step: [`docs/calvin/scene_build_rules.md`](docs/calvin/scene_build_rules.md),
[`docs/calvin/README.md`](docs/calvin/README.md).
