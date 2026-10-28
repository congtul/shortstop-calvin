# Final evaluation (2026-10-08)

1000 `final` scenes x 1 seed, K = 4, freeze, frozen `config.py` parameters,
no intervention-precision rollouts. Outputs: `outputs/calvin_final/`
(`run_final.sh`).

| Method | Safe success | Violation | Timeout | Activation | Fallback | Recovery | Conservatism | Added ms median / p95 |
|---|---|---|---|---|---|---|---|---|
| Unshielded | 0.584 | 0.397 | 0.019 | — | — | — | — | — |
| Conf-Thresh | 0.590 | 0.396 | 0.014 | 0.299 | 0.054 | 0.642 | 0.025 | 23 / 23 |
| STL-Monitor | 0.375 | 0.010 | 0.615 | 0.940 | 0.939 | 0.042 | 0.355 | 34 / 41 |
| CBF | 0.795 | 0.028 | 0.177 | 0.463 | 0.023 | 0.708 | 0.020 | 57 / 166 |
| MPC-Filter | 0.797 | 0.018 | 0.185 | 0.481 | 0.069 | 0.715 | 0.022 | 7 / 914 |
| ShortStop | 0.799 | 0.012 | 0.189 | 0.509 | 0.120 | 0.711 | 0.013 | 41 / 151 |

Paired (same scene), ShortStop minus other, bootstrap 95 % CI, McNemar p:

| vs | Δ safe success | Δ violation |
|---|---|---|
| MPC-Filter | +0.002 [−0.013, +0.017], p 0.90 | −0.006 [−0.015, +0.003], p 0.26 |
| CBF | +0.004 [−0.011, +0.018], p 0.69 | −0.016 [−0.026, −0.007], p 0.002 |
| STL-Monitor | +0.424 [+0.393, +0.455], p < 1e−100 | +0.002 [−0.007, +0.011], p 0.82 |
| Conf-Thresh | +0.209 [+0.181, +0.238], p < 1e−40 | −0.384 [−0.414, −0.354], p < 1e−100 |
| Unshielded | +0.215 [+0.187, +0.243], p < 1e−40 | −0.385 [−0.415, −0.354], p < 1e−100 |

Supported: ShortStop matches MPC-Filter and CBF on success, has fewer
violations than CBF, is as safe as STL-Monitor at +42 points of success, and
has the lowest p95 latency among the correcting shields. Not supported: any
success or violation advantage over MPC-Filter.
