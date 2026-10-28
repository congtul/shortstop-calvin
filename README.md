# ShortStop

Code and data for *Counterexample-Guided Short-Horizon Shielding of
Generative Robot Policies* (ACIIDS 2027 submission, Springer LNCS; source and
PDF: [`docs/paper/shortstop/`](docs/paper/shortstop/)).

ShortStop is a runtime safety filter for a frozen generative policy that
emits action chunks. At every decision it

1. **proposes** K chunks from the policy,
2. **reaches**: propagates a short-horizon reachtube of the whole arm under a
   calibrated per-step model error, R̂_{k+1} = f̂(R̂_k, a_{t+k}) ⊕ B(ε_k + w̄),
3. **certifies** a lower bound on the STL robustness-to-go over the tube,
4. **selects** the certified chunk with the best task progress, or, when none
   certifies, **repairs** one along its worst-case counterexample, and falls
   back to holding the arm only when repair fails.

A certified chunk satisfies the specification over the certified horizon
whenever the tube is sound (Theorem 1); with inference delay d the horizon
must cover H ≥ r + d rows (Remark 1).

## Results on CALVIN

Frozen MDT diffusion policy (`mdtv-1-abcd`, environment D), 1000
pre-registered test scenes with 1-3 obstacles the policy cannot see, one
rollout per scene, same policy seed for every method:

| Method | Safe success % [95% CI] | Violation % | Timeout % | Added latency ms (median / p99) |
|---|---|---|---|---|
| Unshielded | 58.4 [55.3, 61.5] | 39.7 | 1.9 | - |
| Conf-Thresh | 59.2 [56.2, 62.3] | 39.2 | 1.6 | 37 / 71 |
| STL-Monitor | 39.8 [36.8, 42.8] | 0.0 | 60.2 | 48 / 80 |
| MPC-Filter | 78.5 [76.0, 81.0] | 1.0 | 20.5 | 13 / 652 |
| CBF | 80.3 [77.8, 82.8] | 0.7 | 19.0 | 101 / 230 |
| **ShortStop** | 80.0 [77.5, 82.5] | 1.1 | 18.9 | 56 / 328 |

ShortStop matches the barrier and predictive filters that share its model
and margins, beats certification without repair by 40 points, and keeps the
arm inside its certified tube in 99.9 % of executed steps. The paper also
reports an inference-delay stress test, a component ablation and a horizon
sweep.

## Repository

```
shortstop/calvin/        reachtube (reach.py), shields and baselines (shield.py), g(a) (progress.py),
                         closed-loop harness (experiment.py), scene builder, config.py
shortstop/calvin/data/   frozen scenes (40 val, 1000 test) and geometry caches
scripts/                 run, analysis, calibration, scene-building and verification scripts
scripts/pipeline/        the stages that produced the paper's runs (pre_final, final, delay, ablation)
tests/                   unit tests (no simulator needed)
docs/calvin/             protocol and configuration (README.md), lab notebook, audits, run records
docs/paper/shortstop/    paper source (LaTeX, LNCS); tables and Fig. 2 generated from the rollouts
patches/                 the patch applied to the mdt_policy submodule
env/                     pip freeze of the environment of the reported runs
mdt_policy/              submodule: MDT policy and the CALVIN simulator
```

## Getting started

- Check the paper's numbers from the released rollouts (no GPU, a minute):
  [`REPRODUCE.md`](REPRODUCE.md), level 0.
- Install CALVIN, MDT and ShortStop: [`docs/CALVIN_SETUP.md`](docs/CALVIN_SETUP.md).
- Rerun scenes and experiments: [`REPRODUCE.md`](REPRODUCE.md), levels 1-3.
- What exactly was run, and why: [`docs/calvin/README.md`](docs/calvin/README.md).

```bash
pytest tests -q
python scripts/run_calvin_test.py +split=final +seeds=1 +precision=false \
    +baseline=shortstop +scene_id=final_0010
```

## Acknowledgements

Built on [CALVIN](https://github.com/mees/calvin) and
[MDT](https://github.com/intuitive-robots/mdt_policy); their code, checkpoint
and dataset are used unmodified apart from `patches/mdt_policy_shortstop.patch`
and are subject to their own licenses
([`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)).

## License

MIT ([`LICENSE`](LICENSE)) for the ShortStop code and the frozen scenes; the
released data archives (release v1.0) are CC BY 4.0.
