# Fairness audit: what each method can and cannot handle

Written 2026-10-14 after the v2 final, before the delay stress test. An
asymmetry here is reported as a limitation of the method that has it; the
final run is not changed after the fact.

| Situation | ShortStop | MPC-Filter | CBF | STL-Monitor | Conf-Thresh |
|---|---|---|---|---|---|
| Obstacle on the chunk | certify all K; repair (Polyak) if none passes | SQP correction of candidates[0] | per-row QP correction of candidates[0] | reject; pick another of K | not seen (no obstacle knowledge) |
| Model error | per-step ε_k (tube) | ε_k tightening per step | ε_k margin per corrected row | none (no tube, by definition) | none |
| Joint limits | **nominal reject only: repair cannot fix them** -> fallback (`fail_000`, 3 % of decisions, 14 episodes, all failed; CBF solves 4 of them) | QP constraint (corrected) | QP constraint (corrected) | nominal reject | none |
| Arm already inside the margin | no chunk certifies -> hold (may stall) | QP often infeasible -> hold | barrier alpha*h lets it move out (by CBF's design) | reject -> hold | – |
| Fallback | joint hold, anchored | same | same | same | same |
| Look-ahead | H rows, whole chunk | H rows | one row at a time, r rows | H rows | H rows (disagreement) |
| Choice among K | yes | no (candidates[0]) | no (candidates[0]) | yes | yes |
| Delay (aware mode) | committed rows + chunk, H = r + d, committed rows fixed | same rule | corrects rows d..r+d-1 | not in the stress test | not in the stress test |

The joint-limit asymmetry costs ShortStop at most about 0.4 points of safe
success in the final (14 of 1000 episodes); it does not change any paired
result. Safety and Theorem 1 are unaffected (rejecting more never admits an
unsafe chunk).
