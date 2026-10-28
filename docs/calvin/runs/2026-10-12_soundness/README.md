# Certificate soundness (2026-10-12)

ShortStop on the first 300 `final` scenes with `+soundness=true` (passive:
300/300 rollouts identical to the final run). Per executed row k of every
non-fallback decision: certified robustness of tube step k, true robustness
of the reached arm (same primitives, no inflation), max frame deviation
predicted vs reached. 41 043 rows. Output:
`outputs/calvin_final/shortstop_freeze_soundness_smoke.jsonl`.

| k | rows | deviation p50 / p99 / max (mm) | true < certified |
|---|---|---|---|
| 1 | 10 346 | 1.8 / 7.2 / 21.5 | 0.01 % |
| 2 | 10 287 | 3.3 / 13.4 / 41.4 | 0.01 % |
| 3 | 10 232 | 4.7 / 19.7 / 61.0 | 0.15 % |
| 4 | 10 178 | 6.1 / 25.9 / 81.0 | 0.57 % |
| all | 41 043 | | 0.18 % (75 rows) |

`MODEL_ERROR` = 14.7 mm is exceeded by 3.6 % of rows overall.

Finding: the deviation grows about linearly with k, but the implemented tube
inflates every step by the same one-step `MODEL_ERROR` (`reach.propagate_arm_tube`).
The paper's Eq. (1) propagates the set and adds B(ε_k + w̄) at every step,
so its radius grows with k (Eq. (6): r_k ≤ r_0 + k(ε + w̄) for L ≤ 1). The
implementation is therefore not the paper's tube for k ≥ 2; robustness stays
empirically sound in 99.8 % of rows because the worst frames are rarely the
closest primitive. All model-based shields share the same tube/tightening,
so the comparison stays fair; the soundness claim does not hold as stated.
