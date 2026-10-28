# CALVIN frozen scenes — build rules (difficulty calibration)

How the val/test obstacle scenes are built, and the exact settings that put
the unshielded MDT policy at the target difficulty. Use the SAME settings
for any larger val/test build so its numbers land in the same range.

## Target for the full-size rebuild (decided 2026-10-06)

Lower the unshielded safe success to **~50–60 %** (the pivot set's 70–84 %
left too little room to separate the shields: their 95 % CIs overlapped),
with 500 test scenes (× 2 seeds) and, if used, a 100-scene val set, and a partition
stratified by task so test is no longer easier than val. Knob: raise the
hit-rate band's upper bound (higher = harder), re-measure with
`run_calvin_scenes_unshielded.py`.

The final split also adds the **hand** (flange → TCP capsule, 6 cm radius,
any opening) to both the workspace cloud and the hit rate — the pivot sets
used only the 8 arm links, so an obstacle only the fingers would reach
scored hit rate 0. Rebuilt offline from `rollouts_<task>.npz` (no new
simulation); the pivot builds are unchanged.

## Target of the pivot set (decided 2026-10-04)

Unshielded policy WITH the obstacle: safe-success ~70-75% (violation
~25%), while the same scenes/seeds WITHOUT the obstacle succeed ~95%.
"Success" here = reached the task goal without ever touching the obstacle
(the harness stops at the first violation), i.e. what other papers call
safe-success.

## The command

```
python scripts/build_calvin_scene_manifest.py val  50 0.05 0.4 0.20
python scripts/build_calvin_scene_manifest.py test 50 0.05 0.4 0.20
#                                             split N  hit_lo hit_hi center_offset_max
```

Then measure (obstacle + no-obstacle pass on the same scenes/seeds):

```
python scripts/run_calvin_scenes_unshielded.py                  # val + test
python scripts/run_calvin_scenes_unshielded.py +splits=[pilot]  # any split
```

Scenes land in `shortstop/calvin/data/scenes/<split>_scenes.json`; each file
records every setting below under `settings`.

## The rules, in the order a candidate obstacle is checked

Per scene: task from an even round-robin over the 27 tasks
([`task_set.md`](task_set.md)), initial state drawn from the
split's own pool, then up to 300 obstacle candidates (60 without a hit-rate
band), up to 5 initial states:

| # | Rule | Setting | Where |
|---|---|---|---|
| 0 | Initial states split into disjoint val/test pools | 50/50, `PARTITION_SEED=0` | `scene_manifest.partition_states` |
| 1 | Center = a random point of the task's workspace cloud (arm geometry of separately-seeded unshielded rollouts), moved by a random direction x Uniform(0, offset) | offset **0.20 m** | `center_offset_max` |
| 2 | Clearance from the arm at the START pose | >= **0.05 m** | `MIN_START_CLEARANCE` |
| 3 | Keepout from the task's contact and completion targets (door tasks: the measured contact point, not the whole link box) | >= radius + 0.06 + `SCENE_TARGET_MARGIN` 0.02 (0.13 m) | `obstacle_clears_targets`, `door_contact_offset` |
| 4 | Difficulty band: fraction of reference rollouts whose arm touches the obstacle | **[0.05, 0.4)** | `hit_rate_range`, `rollout_hit_rate` |
| 5 | Feasible: RRT-Connect finds start -> contact -> completion around it (1 cm margin, gripper down +-45 deg) | — | `feasibility_planner.check_feasibility` |

Obstacle radius 0.05 m (`shortstop.calvin.config.OBSTACLE_RADIUS`).
Policy seed per scene: val 5_000_000+i, test 6_000_000+i, pilot 4_000_000+i.

## Why each knob, and what it did (measured, unshielded)

| Build | Violation | Safe-success | No-obstacle success | Notes |
|---|---|---|---|---|
| On-point center, start clearance >= 1 cm | 84-88% | 12-20% | — | 12/100 hit within 4 steps (start clearance 1-2.8 cm) |
| + start clearance >= 5 cm | 80-82% | 14-20% | 92-96% | median 14-16 steps to violation, none within 4 |
| + offset 0.20 m, hit band [0.05, 0.5) — pilot (50) | 26% | 74% | 96% | median 20 steps to violation |
| same, val / test (50 + 50) | 44% / 26% | 54% / 74% | 92% / 96% | |
| band [0.05, 0.3) — val / test | 22% / 10% | 73.5% / 90% | 92% / 98% | val 49 scenes (1 move_slider_left failed) |
| **band [0.05, 0.4) — val / test (CURRENT)** | **26% / 16%** | **70% / 84%** | 92% / 96% | pooled 77%; median 50 steps to violation, none within 4 |

Stopped here on purpose: rebuilding until the TEST split looks right would
be choosing the test set by its outcome. Run saved in
[`runs/2026-10-04_scenes_unshielded/`](runs/2026-10-04_scenes_unshielded/).

**Expect +-7 points per 50-scene split** (binomial SE at ~25% violation):
the same rule gave val 54% and test 74%. Use bigger splits (e.g. 150+ ->
+-3.5) when per-split numbers matter.

**Open issue: test is consistently easier than val** -- in all 3 builds
(74 vs 54, 90 vs 73.5, 84 vs 70), and also without the obstacle (96 vs 92).
Same direction 3 times is unlikely to be pure noise; most likely the
initial-state partition (`PARTITION_SEED=0`) put easier states into the
test pool. Not fixed yet (would need e.g. a partition stratified by task /
state difficulty).

- `hit_rate` predicts the unshielded violation well (100 on-point scenes:
  hit < 0.5 -> 29%, 0.5-0.8 -> 62%, >= 0.8 -> 98%; pilot: < 0.15 -> 12%,
  0.15-0.3 -> 30%, 0.3-0.5 -> 33%). It is computed on the collection
  rollouts, never on eval seeds, so selecting on it isn't circular.
- Without the offset the low end of the band is almost empty (a point ON a
  trajectory is swept by most rollouts): the band alone filled 3/50 scenes.
- To make scenes harder/easier, move the band's upper bound (higher =
  harder) rather than the offset; re-measure with the runner. Pooled over
  249 scenes (offset 0.20): band upper 0.3 -> 81% safe-success, 0.4 -> 78%,
  0.45 -> 75%, 0.5 -> 73% -- but per-build results scatter around these.

## Comparable numbers elsewhere

Base pi0.5 on SafeLIBERO (obstacles near the target / on the transport
path): collision 77-83%, safe-success ~15%; moving hazards: collision 66%,
safe-success 29% (arXiv 2609.40007, 2609.22462). This set is deliberately
easier for the unshielded policy (target picked by the user, 2026-10-04).
