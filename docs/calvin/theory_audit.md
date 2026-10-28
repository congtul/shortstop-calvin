# Theory vs code audit (2026-10-12)

Status after the v2 fixes (commit fde7844) at the end of this file.

Paper: the early draft (removed; git tag `prototype-2d`), Sec. III-IV, App. A-B;
the current paper is `docs/paper/shortstop/main.tex`.
Code: `shortstop/calvin/{reach,shield,experiment,config}.py`.

## A. Breaks a stated guarantee (code fix + rerun)

1. **Tube radius does not grow (Eq. 1, Assumption 1).** Eq. (1) adds
   B(ε_k + w̄) at every step, so the radius grows with k (Eq. 6).
   `propagate_arm_tube` propagates the nominal path and inflates every step
   by the same one-step `MODEL_ERROR`. Measured (runs/2026-10-12_soundness):
   frame deviation p99 7 / 13 / 20 / 26 mm at k = 1..4 vs a fixed 14.7 mm;
   3.6 % of executed rows outside the tube. MPC/CBF tightening shares it.

## B. Assumptions the implementation does not establish

2. **ε is a quantile, not a bound.** p99 x 1.25 of one-step residuals on
   unshielded obstacle-free rollouts (also the draft's own recipe): Theorem 1
   then holds with high probability, not surely. State it so.
3. **Hidden state.** CALVIN's IK tracks an accumulated target pose; the arm
   trails it (~4 mm, env_verification.md). x_t = joint angles omits that
   offset, so the true next state depends on more than (x_t, a_t); the
   offset accumulates over a chunk (one cause of finding 1).
4. **Fallback invariance (Assumption 3, Corollary 1) unverified.** Freeze =
   zero relative action; the arm keeps converging to the IK target after
   it, and ShortStop's margin is 0, so "freeze keeps x in X_s" is not shown.
5. **Joint limits on the nominal path only** (`_trajectory_within_joint_limits`),
   not on the inflated tube.

## C. Paper text / algorithm differs from the code (theory unaffected)

6. Execution: paper commits the first action; code runs REPLAN_STEPS = 4
   certified rows (inside H, so Theorem 1 holds; Corollary 1's induction is
   per decision).
7. Repair: paper = one step a + ηd in an l2 trust region δ, applied to every
   failing candidate; code = Polyak (re-linearized, <= 5 rounds/candidate,
   per-row cap 10 cm), only when no candidate certifies. Prop. 1's cost term
   c_rep δ must use the row cap.
8. Counterexample: paper = M = 16 projected-gradient steps over zonotopes;
   code = exact argmin over capsule primitives (closed form). Tube = capsules,
   not intervals/zonotopes.
9. Spec: paper = obstacles + table/self-collision + workspace/joint limits +
   progress (eventually); code = obstacles + joint limits; progress is g(a).
10. Margin: paper ε = 2 cm; code 0 (ShortStop, MPC, CBF), STL-Monitor 1 cm
    (paper: STL-Monitor rejects only negative robustness).
11. Table VII: K 8 -> 4, w̄ 0.5 cm -> 0, δ/η unused, 500 x 5 -> 1000 x 1, H = 8.
12. Eq. (2) is over k = 0..H; code k = 1..H (k = 0 is the measured state).
13. Prop. 1 / Fig. 4 use r_H from Eq. (6); with finding 1 the code's radius
    is constant in H, so the H-sweep would not test Eq. (6) until fixed.
14. Discrete time: shields and ground truth both check env steps only
    (~1.5 cm of motion per step is not swept).

## Checked and consistent

- Shield primitives cover the real fingers (300 random configs, widths
  0-8 cm: worst uncovered depth 0.0 mm).
- A repaired chunk is certified with the same tube before it runs.
- MPC-Filter/CBF/ShortStop share f-hat, ε and fallback (paper Sec. V-D).
- Action scale, determinism (env_verification.md).

## v2 fixes (2026-10-12)

| Item | Fix | Check |
|---|---|---|
| 1 tube radius | per-step MODEL_ERROR (p99 x 1.25 of the k-step error), 9.6 -> 36.7 mm over k = 1..8 | soundness on 12 val scenes (pipeline gate) and on 300 test scenes |
| f-hat | pose model: position + orientation columns, damped least squares (wrist singularities); shared by ShortStop, MPC-Filter, CBF, STL-Monitor, Conf-Thresh and g(a) | k = 8 p99 29 mm vs 59 mm (position-only) |
| 3 hidden state | absorbed: MODEL_ERROR is measured on real k-step motion | as item 1 |
| 4 fallback | joint hold (joints keep their angles, IK target reset): drift p99 1.1 mm over 8 steps (freeze 17 mm, Cartesian brake 10 mm with null-space motion); certified margin 3 mm for ShortStop/MPC/CBF | calibration run_20261012_* |
| 5 joint limits | out of the certified spec (nominal filter); the pose model's joint error is 0.05 -> 0.44 rad | text: spec = obstacle avoidance |
| 2 ε is a quantile | unchanged: Theorem 1 holds with high probability; report measured soundness | — |

Hold anchor (found during the first v2 final, which was stopped): re-reading
the current angles at every hold step let the drift compound (~0.13 mm/step
p99), and stalls of 163-307 steps occur, so Assumption 3 failed on long
stalls. Consecutive hold steps now keep the angles of the stall's first
step: whole-arm drift max 0.25 mm over 300 steps (0.245 mm; largest per-step
p99 0.23 mm; brake: 31 mm), below the
3 mm margin for any observed stall (run_20261012_164600, FREEZE_PROBE_STEPS=300,
3 sequences). Every v2 run before this fix is archived and rerun.
