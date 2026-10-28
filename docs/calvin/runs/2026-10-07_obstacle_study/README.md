# Obstacle-layout study (2026-10-07, overnight)

Question: with the tuned parameters frozen and the freeze fallback, does a
more complex obstacle layout (where QP filters lose their convex,
single-constraint advantage) let ShortStop's heuristic repair beat
MPC-Filter / CBF?

## 1. Sim probes (CALVIN + MDT, freeze, 1 seed, val-pool states)

Families in `scene_manifest.FAMILIES`; safe success (all shields: 0 % violation).

| Family (scenes) | Unshielded | ShortStop | STL-Monitor | MPC-Filter | CBF | MPC/CBF infeasible or solver error | MPC p95 select |
|---|---|---|---|---|---|---|---|
| clutter3 (5) | 60 % | 40 % | 40 % | 60 % | 60 % | 44 % / 41 % of decisions | 1.6 s |
| near_target (10) | 70 % | 80 % | 60 % | 90 % | 90 % | 0 / 0 | 0.56 s |
| bar (10) | 80 % | 90 % | 60 % | 90 % | 90 % | 0 / 0 | 1.2 s |
| mixed (8) | 75 % | 62 % | 50 % | 75 % | 75 % | 0 / 0 | 2.3 s |

Too few scenes for rankings (±15–20 points); the families are still easy.

## 2. One-decision study (`one_step.py`, no simulator)

101 real arm configurations, a nominal chunk 1.5 cm/row that is certain to
hit the layout 14 cm ahead, 8 near-identical candidates. "Moving" = a
certified safe chunk that still advances > 5 mm.

| Layout (spheres) | safe & moving: STL / ShortStop / MPC / CBF | mean select time: STL / ShortStop / MPC / CBF |
|---|---|---|
| sphere (1) | 0 / 0 / 1.00 / 1.00 | 39 ms / 118 ms / 323 ms / 55 ms |
| bar (5) | 0 / 0.91 / 0.15 (backs off) / 1.00 | 67 ms / 215 ms / 1.1 s / 182 ms |
| box (70) | 0 / 0.41 / 0.30 (backs off) / 1.00 | 0.5 s / 1.9 s / 16.6 s / 2.4 s |
| corner (84) | 0 / 0.95 / 0.99 / 1.00 | 0.6 s / 1.8 s / 26 s / 2.9 s |
| clutter3 (44) | 0 / 0.33 / 0.94 / 1.00 | 0.3 s / 1.3 s / 11 s / 1.5 s |

CBF is checked on the 4 executed rows (its own scope); the others on 8.

## 3. Repair diagnostic (`repair_sensitivity.py`, not tuning)

ShortStop safe-and-moving rate / mean time, by trust region and repair iterations:

| Layout | 3 cm / 1 it | 3 cm / 3 it | 5 cm / 1 it | 5 cm / 3 it | 10 cm / 1 it | 10 cm / 3 it |
|---|---|---|---|---|---|---|
| sphere | 0.00 | 0.00 | **0.88** | 0.88 | 0.00 | 0.00 |
| bar | **0.91** | 0.96 | 0.00 | 0.00 | 0.00 | 0.01 |
| box | 0.41 | **0.87** | 0.41 | 0.38 | 0.01 | 0.00 |
| corner | 0.95 | 0.99 | 0.89 | 0.89 | 0.44 | 0.48 |
| clutter3 | 0.33 | 0.45 | **0.79** | 0.79 | 0.00 | 0.00 |

## Findings

1. **QP cost explodes with obstacle complexity**: MPC-Filter goes from 0.3 s
   (one sphere) to 11–26 s per decision with box-like or cluttered layouts,
   and in the sim clutter probe ~44 % of its decisions are infeasible.
   ShortStop stays ~1–2 s (pure-Python distances; vectorizable).
2. **CBF is the strongest filter here**: one-step QPs stay small, always
   feasible offline, and keep moving.
3. **ShortStop's repair, as specified (one fixed-size gradient step clipped
   to a trust region), is brittle, not weak**: the success rate jumps between
   0 and 0.9 depending on whether that single step lands in free space
   (sphere solves at 5 cm, bar only at 3 cm, 10 cm fails everywhere). The
   parameters tuned on the single-sphere pivot do not transfer to other
   layouts.
4. Implication: the heuristic-vs-QP trade-off is real on compute (vs MPC),
   but ShortStop needs a robust repair step to also win on solve rate — e.g.
   trying a few step sizes inside the trust region and keeping the first
   certified chunk (a line search; no new tuned parameter). This changes the
   method and needs the author's decision.
