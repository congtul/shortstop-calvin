# Local-cluster val (2026-10-07)

40 `probe_local_cluster` scenes x 1 seed, K = 4, freeze, ep_len 360. Tuning
grids in `baseline_runner.GRIDS`; outputs in `outputs/calvin_tuning_probe_local_cluster/`
and `outputs/calvin_probe_local_cluster/` (`*_K4`).

## Tuning

| Baseline | Config | Safe success | Violation | Timeout |
|---|---|---|---|---|
| STL-Monitor | ε = 0 | 0.450 | 0.125 | 0.425 |
| | **ε = 0.01** (chosen: safety) | 0.325 | 0.025 | 0.650 |
| | ε = 0.02 | 0.300 | 0.000 | 0.700 |
| CBF | α = 0.25 | 0.775 | 0.150 | 0.075 |
| | **α = 0.5** | 0.775 | 0.125 | 0.100 |
| Conf-Thresh | 2 mm | 0.575 | 0.400 | 0.025 |
| | **3 mm** | 0.550 | 0.425 | 0.025 |
| | 5 mm | 0.500 | 0.475 | 0.025 |

Selection rule (`run_calvin_tuning.py`): best safe success, ties within
`SELECTION_TOLERANCE` broken by lower activation; STL-Monitor overridden to
ε = 0.01 by choice.

## All methods (chosen parameters)

| Method | Safe success | Violation | Timeout | Fallback | Select ms mean / p95 / p99 / max |
|---|---|---|---|---|---|
| Unshielded | 0.525 | 0.450 | 0.025 | — | — |
| Conf-Thresh | 0.550 | 0.425 | 0.025 | 0.056 | 19 / 20 / 21 / 23 |
| STL-Monitor | 0.325 | 0.025 | 0.650 | 0.938 | 31 / 37 / 38 / 65 |
| CBF | 0.775 | 0.125 | 0.100 | 0 | 83 / 166 / 228 / 243 |
| MPC-Filter | 0.825 | 0.100 | 0.075 | 0.006 | 126 / 505 / 806 / 2910 |
| ShortStop | 0.775 | 0.100 | 0.125 | 0.241 | 73 / 237 / 250 / 390 |

Latency without each rollout's first decision (solver warm-up, fixed since).
Paired per scene: ShortStop vs CBF 2/2 discordant, vs MPC 0/2, vs STL 19/1.

## Design checks on the first 12 scenes

- Repair: Polyak 0.75 = MPC = CBF; the paper's trust-region step 0.50.
- Rounds per candidate: c = 20 rarely falls back but has a 2.7 s tail; c = 5
  bounds the worst case and stalls in scene 0006 (fail -> freeze loop), same
  outcome. c = 5 chosen.
- Repair margin 3 vs 1 mm: no difference in outcome; 1 mm = `SQP_MARGIN`.
- K = 1 / 4 / 8: 0.75 / 0.67 / 0.75 (one scene apart), latency ~linear in K
  (`shortstop_freeze_c5_K1`, `_K4`, and `shortstop_freeze_c5` = K 8, the
  default then; this configuration: freeze fallback, margin 0, one constant
  model error, superseded by the final one).
- Intervention precision: with the privileged rollout on, 7/10 Conf-Thresh
  scenes are identical to the run without it (10/10 between two runs without
  it) -> dropped.
