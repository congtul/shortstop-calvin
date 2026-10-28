# Repair study (2026-10-07)

Scenes: `probe_local_cluster` (1-3 separate spheres, 3-7 cm, within 20 cm of
one anchor near the goal or on the path, surface gap >= 2 cm, RRT with a 2 cm
margin; 12 scenes, val-pool states). Freeze fallback, seed 0, pivot params.

## Why the paper's repair stalls (scene 0010, push_into_drawer)

ShortStop (paper repair: one step a + eta d, l2 trust region on the whole
chunk) timed out with 300 fallbacks while MPC-Filter / CBF went around and
succeeded. On the 40 captured failing decisions (`capture_*.npz`) the worst
violation was the wrist (frame 6) at step 8, ~2 cm deep. The trust region
spreads delta = 3 cm over 8 rows (1.06 cm each), mapped to the wrist through
the TCP Jacobian -- never enough; eta is irrelevant because the trust region
always binds. Offline on those 40 decisions:

| Repair | solved | forward progress (4 rows) | time |
|---|---|---|---|
| paper (gradient, trust region) | 0 % | -- | 146 ms |
| all-violations direction, fixed 2 cm steps | 100 % | 2.4 cm | 696 ms |
| **Polyak: eta = (eps + 3 mm - rho) / |grad|^2, iterated** | 100 % | 2.9 cm | 501 ms |
| counterexample QP (min-norm over all violated steps) | 100 % | 2.9 cm | 307 ms |
| MPC-Filter / CBF | 100 % | 3.1 / 4.1 cm | 436 / 89 ms |

## Closed loop (sim), all 12 scenes

| Method | safe success | violation | timeout | fallback | steps | select ms mean / p95 |
|---|---|---|---|---|---|---|
| Unshielded | 58 % | 42 % | 0 % | -- | 58 | -- |
| ShortStop, paper repair | 50 % | 8 % | 42 % | 53 % | 208 | 103 / 151 |
| **ShortStop, Polyak repair** | **75 %** | 8 % | 17 % | 2 % | 149 | 169 / 378 |
| ShortStop, counterexample QP | 67 % | 8 % | 25 % | 41 % | 164 | 529 / 1086 |
| STL-Monitor | 25 % | 0 % | 75 % | 96 % | 291 | 44 / 55 |
| MPC-Filter | 75 % | 8 % | 17 % | 0 % | 136 | 132 / 501 |
| CBF | 75 % | 8 % | 17 % | 0 % | 134 | 90 / 190 |

Per scene, Polyak-repair ShortStop now succeeds exactly where MPC-Filter and
CBF do (scenes 0002, 0008, 0010 recovered); all fail 0000/0006 (timeout) and
0011 (collision, every shield).

Polyak repair keeps the paper's form a' = a + eta d but computes eta from the
robustness deficit (the minimum-norm correction of the linearized worst
constraint), re-linearizes until the tube certifies (<= 20 steps), and keeps
only a 10 cm per-row cap instead of the chunk-level trust region. No new
tuned parameter. Implemented as `ArmRepairShield(repair_mode="polyak")`.
