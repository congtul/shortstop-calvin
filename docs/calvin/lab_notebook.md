# ShortStop on CALVIN: lab notebook

Chronological record of the CALVIN study: pilots, superseded settings (K = 8,
H = 8, freeze fallback, one constant model error), the decisions taken and
why, and the registrations made before each long run. Sections near the top
describe earlier protocols and are kept as they were written; the
configuration of the reported runs is [`README.md`](README.md) and the
section "Final v2" below. Paths `outputs/pipeline_v2/*.sh` are now
`scripts/pipeline/*.sh`.

The policy is the MDT checkpoint `mdtv-1-abcd` (ABCD→D) run in environment D;
earlier mentions of "task_D_D" refer to the environment layout, not to the
checkpoint's training split.

## Protocol

- **One task per rollout** from one of CALVIN's fresh initial states, up to
  `ep_len = 360` steps. A rollout ends at success (CALVIN's own task
  oracle), at the first collision with the obstacle, or at the step limit.
  *Safe success* = success without any collision.
- **27 tasks**: every CALVIN task that can start from a fresh state, minus
  stack/unstack — see [`task_set.md`](task_set.md).
- **Frozen scenes** (task, initial state, obstacles, policy seed), built
  offline. Final protocol: `local_cluster` obstacles (1-3 spheres around one
  anchor, see below); tuning val = `probe_local_cluster` (40 scenes,
  unshielded safe success 52.5 %), test = `final` (1000 scenes, 97 initial
  states disjoint from every val/pivot scene). The earlier single-sphere
  pivot sets (50 val + 50 test) are kept for reproducibility only. Rules:
  [`scene_build_rules.md`](scene_build_rules.md).
- **Tuning on val only**, test run once at the end, 1 seed (policy frozen, so
  the variance is across scenes; same scenes and seed for every method).

## Decision loop (P-R-C-S)

Every `REPLAN_STEPS = 4` env steps:

1. **Propose** — MDT samples `N_CANDIDATES = 8` chunks of 10 relative actions
   (TCP Δposition, ΔEuler, gripper), in one batched call
   (`shortstop/calvin/mdt_policy_client.py`).
2. **Reach** — each chunk's first `HORIZON_STEPS = 8` rows are turned into a
   capsule reachtube of the whole arm: TCP moves (`action × 0.02 m`, CALVIN's
   `max_rel_pos`, checked in sim) → joint steps via the pseudo-inverse of the
   TCP Jacobian → forward kinematics. Capsules are inflated by
   `model_error + w_bar` (`shortstop/calvin/reach.py`).
3. **Certify** — STL robustness-to-go = smallest capsule–obstacle distance
   over the horizon; joint limits checked too (`shortstop/calvin/shield.py`).
4. **Select / Repair** — the admissible candidate with the best task-progress
   score g(a); ShortStop repairs a failing candidate along the counterexample
   direction; with nothing admissible the **freeze fallback** holds the arm
   for one step and proposes again (retreat — a geometric push away from the
   nearest obstacle — is an ablation; see "Fallback" below).

Ground truth for collisions (`experiment._clearance`) uses the 8 link
capsules and the two real finger capsules at the current gripper width.

### Task-progress score g(a)

`g(a) = −distance(predicted TCP after the chunk, current goal)`
(`shortstop/calvin/progress.py`). Goals come from
`shortstop/calvin/task_goals.py`, the same definition the scene builder's
feasibility planner proves reachable:

| Task kind | Contact goal (phase 1) | Completion goal (phase 2) |
|---|---|---|
| push block | block center | block moved by the task's (dx, dy) |
| lift block | block center | block raised by dz + 2 cm |
| rotate block | block center | (wrist turn — not scored, keeps contact goal) |
| drawer / slider | measured fingertip contact point on the door | same point with the door moved by its threshold |
| switch / button | actuator box at its current value | actuator at the other end / pressed |
| push into drawer | nearest table block | drawer box |

Phase 2 starts (and stays) once the object has moved: a block by > 1 cm, a
joint by > 10 % of its travel. Door contact points:
`shortstop/calvin/data/door_contact_offsets.json`.

**g(a) on vs off** (val, 50 scenes, same seeds; `run_calvin_g_a_check.py`,
[`runs/2026-10-05_g_a_check/`](runs/2026-10-05_g_a_check/)):

| Vehicle | safe success on / off | violation on / off | discordant pairs (only on / only off) |
|---|---|---|---|
| Best-of-N | 70 % / 72 % | 28 % / 26 % | 0 / 1 |
| STL-Monitor | 58 % / 62 % | 14 % / 12 % | 1 / 3 |

Rollouts do differ (step counts differ in 45/50 pairs) but outcomes almost
never: picking among the policy's own 8 samples by g(a) is neutral on these
scenes. g(a) reached phase 2 in 34/35 Best-of-N successes.

### Baselines

| Name | Class | Uses g(a) | Notes |
|---|---|---|---|
| Unshielded | — | no | executes `candidates[0]` |
| Best-of-N | `ArmConfThreshShield(threshold=∞)` | yes | g(a) only, no safety filter |
| Conf-Thresh | `ArmConfThreshShield` | yes | rejects on TCP disagreement across the K candidates |
| STL-Monitor | `ArmSTLMonitorShield` | yes | nominal STL robustness, no tube inflation, no repair |
| MPC-Filter | `ArmMPCFilterShield` | no | predictive safety filter: H-step OCP (min deviation from `candidates[0]`, tightened collision + joint constraints) solved by SQP |
| CBF-Shield | `ArmCBFShield` | no | per-step CBF QP on each executed row along f-hat |
| ShortStop | `ArmRepairShield` | yes | tube + STL certify; only when no candidate certifies, Polyak repair of the worst counterexample (closed form, re-linearized, at most 5 rounds per candidate), candidates tried in g(a) order |

## Parameters (`shortstop/calvin/config.py`)

| Parameter | Value | Source |
|---|---|---|
| `N_CANDIDATES` | 4 | K ablation 1/4/8 on val: no gain from more MDT samples (mm spread); ShortStop's worst-case latency scales with K |
| `CHUNK_LENGTH` / `REPLAN_STEPS` / `HORIZON_STEPS` | 10 / 4 / 8 | checkpoint / design / paper default H |
| `OBSTACLE_RADIUS` | 0.05 m | scenes |
| `MODEL_ERROR` | 0.01473 m | p99 × 1.25 of the one-step prediction error of the TCP-Jacobian model, 7036 real steps (`scripts/calibrate_arm_model_error.py`) |
| `W_BAR` | 0 | `env.step` is deterministic ([`env_verification.md`](env_verification.md)) |
| `SCENE_TARGET_MARGIN` | 0.02 m | scene builder keepout (fixed with the frozen scenes) |
| `STL_MONITOR_EPSILON` | 0.01 m | local_cluster val; chosen over 0 (higher success, 5x the violations): a safety filter trades success for safety |
| `CBF_ALPHA` | 0.5 | local_cluster val (tied with 0.25, fewer violations) |
| `CONF_THRESH_THRESHOLD` | 3 mm | local_cluster val (tied with 2 mm, lower activation) |
| `SHORTSTOP_PARAMS` | eps 0, Polyak, margin 1 mm (= `SQP_MARGIN`), 5 rounds | nothing tuned |

## Scripts, in pipeline order

| Step | Script | Output |
|---|---|---|
| static furniture boxes | `precompute_static_target_sweeps.py` | `shortstop/calvin/data/calvin_task_D_D_target_sweeps.json` |
| reference rollouts | `collect_calvin_unshielded_rollouts.py` | `outputs/unshielded_rollouts/` |
| door contact points | `build_door_contact_offsets.py` | `shortstop/calvin/data/door_contact_offsets.json` |
| scenes | `build_calvin_scene_manifest.py` | `shortstop/calvin/data/scenes/{val,test}_scenes.json` |
| model error | `calibrate_arm_model_error.py` | printed → `config.MODEL_ERROR` |
| unshielded reference | `run_calvin_scenes_unshielded.py` | `outputs/calvin_scenes_unshielded_runs/` |
| g(a) on/off check | `run_calvin_g_a_check.py` | `outputs/calvin_g_a_check/` |
| tune a baseline on val | `run_calvin_tuning.py +baseline=stl_monitor` | `outputs/calvin_tuning/` |
| final test run | `run_calvin_test.py +baseline=stl_monitor +seeds=3 +params_from=...` | `outputs/calvin_test/` |
| render one scene | `render_obstacle_video.py +scene_id=val_0012 +baseline=stl_monitor` | `outputs/render_obstacle_video/` |
| checks | `verify_*.py`, `smoke_test_calvin.py` | printed |

Long runs append one JSON line per rollout and skip finished ones on restart.
No videos are recorded during runs; render chosen scenes afterwards.

## Findings that shape the protocol

**The 8 candidates are nearly identical.** Their predicted TCPs after 8 steps
sit 1.6 mm from their centroid at the median (p90 4.6 mm, p99 9.1 mm, val).
(Remeasured 2026-10-27 in the final configuration, K = 4 candidates, TCP
after H = 4 rows, 40 val scenes, `conf_thresh_hold_spread`: median 0.74 mm,
p90 2.13 mm, p99 6.69 mm; the paper quotes this value.)
Filtering shields therefore accept or reject all candidates together, g(a)
has little to choose from, and Conf-Thresh's thresholds live in millimeters.

**Fallback: retreat, not freeze.** With freeze (zero action, propose again)
a rejected state does not change, the policy proposes the same chunk, and a
filtering shield stays stuck until the step limit. Val, 50 scenes, paired:

| Config | Freeze safe success / timeout | Retreat 2.5 cm safe success / timeout | pairs only retreat / only freeze |
|---|---|---|---|
| STL-Monitor ε = 0 | 60 % / 26 % | 80 % / 6 % | 11 / 1 |
| STL-Monitor ε = 0.01 | 38 % / 60 % | 90 % / 6 % | 26 / 0 |
| ShortStop (untuned) | 84 % / 14 % | 84 % / 12 % | 0 / 0 |

**Decision (2026-10-06): freeze is the default fallback** (code default too).
Retreat is a hand-designed repair (it reads the obstacle position and pushes
the arm away): with it, "STL-Monitor + retreat" is effectively a geometric
repair shield, which a single convex sphere favours. It is kept as an
ablation. The pivot runs below used retreat. Step: 2 cm, one maximal policy action (|a| = 1); 2.5 cm
was no better (84.7 % vs 84.0 % pooled safe success over the three configs).

**MPC-Filter / CBF use the reachtube's geometry.** They used to model the hand
as one 20 cm sphere around the flange, far fatter than the reachtube's flange
point + flange→TCP capsule, so they intervened at ~90 % of decisions. They now
use the same primitives and inflation as ShortStop. With that tighter
geometry one linearization is not enough (the corrected chunk's true tube
still overlapped the obstacle by ~4 cm), so MPC-Filter's optimal control
problem is solved by SQP (re-linearize, up to 5 QPs, 1 mm QP margin, stop at
the first iterate satisfying the nonlinear constraints); no feasible iterate
→ total fallback. Sharing f̂, the collision primitives and the model-error
tightening is the paper's fairness rule, not part of ShortStop's method
(reachtube + STL robustness + counterexample repair over K candidates).
CBF stays the textbook one-step QP.

## Pivot results (50 + 50 scenes)

[`runs/2026-10-06_pivot_stl/`](runs/2026-10-06_pivot_stl/). STL-Monitor val
tuning (retreat 2 cm): ε = 0 → 80 %, **0.01 → 90.7 %** (3 seeds), 0.02 →
91.3 % (3 seeds, higher activation), 0.04 → 82 % (timeouts). Chosen ε = 0.01.

MPC-Filter (nothing to tune: by definition its constraints are distance ≥ 0
after the shared model-error tightening, no ε margin) on val: 92 % safe success; QP status per
decision 74 % unchanged, 25 % corrected, 1.7 % SQP not converged. CBF val
tuning (per-step, retreat): α = 0.1 → 90 %, 0.25 → 92 %, 0.5 → 92.7 %
(3 seeds), 1.0 → 92.0 % (3 seeds). Chosen α = 0.5: the best safe success,
and α = 1 (h_next ≥ 0) degenerates the CBF into a one-step safety filter
with no barrier decay; the activation tie-break would have picked it only
because a larger α intervenes later by construction. The pivot test below
ran with α = 1.0 and is not rerun under retreat; the final (freeze) runs use
α = 0.5.
[`runs/2026-10-06_pivot_mpc_cbf/`](runs/2026-10-06_pivot_mpc_cbf/).

Test, 50 scenes × 3 seeds:

| | Unshielded | STL-Monitor ε = 0.01 | MPC-Filter | CBF α = 1.0 (superseded) |
|---|---|---|---|---|
| Safe success | 83.3 % [77.3, 89.3] | 93.3 % [89.3, 97.3] | 90.0 % [84.7, 94.7] | 89.3 % [84.0, 94.0] |
| Violation | 15.3 % | 0.0 % | 2.0 % | 2.0 % |
| Timeout | 1.3 % | 6.7 % | 8.0 % | 8.7 % |
| Shield activation / fallback | — | 14.9 % / 12.0 % | 14.8 % / 1.7 % | 13.8 % / 1.2 % |
| Intervention precision | — | 0.15 | 0.16 | 0.18 |
| Recovery | — | 0.84 | 0.81 | 0.80 |
| Conservatism cost | — | 3.1 pts | 3.9 pts | 4.7 pts |
| Shield latency (ms) mean / p95 | — | 40 / 42 | 57 / 324 | 58 / 58 |
| Infeasible decisions | — | — | 1.7 % (49 not converged, 15 QP infeasible) | 1.1 % (QP infeasible) |

MPC-Filter and CBF both collide in one scene (`test_0017`, move_slider_left,
all 3 seeds); STL-Monitor never collides. MPC's latency is low on average
because most decisions need no QP; its p95 is the SQP cost.
Brackets: bootstrap 95 % CI over rollouts. Low intervention precision: most
rejected chunks would not actually have collided (the ε margin plus identical
candidates make rejection all-or-nothing).

## Decisions for the final run

- **Fallback: freeze** (zero action for one step, then propose again);
  retreat is an ablation (it gives rejection shields a free recovery reflex).
- **Obstacles: `local_cluster`** (both SafeLIBERO levels: anchor near the
  target or on the policy's path), harder than the pivot sets; final uses the
  val rules unchanged, no difficulty recalibration.
- **Repair: Polyak** (`repair_mode="polyak"`): the paper's single
  trust-region step stalls ([`runs/2026-10-07_repair_study/`](runs/2026-10-07_repair_study/)).
- **K = 4**, ShortStop at most 5 re-linearizations per candidate (real-time
  bound, ~270 ms worst case); repair only when no candidate certifies.
- **Final: 1000 scenes x 1 seed**, bootstrap CIs and paired comparisons
  (same scene and seed across methods).

## Metrics (paper Sec. V.E, one definition for every baseline)

- **Safe success / violation / timeout** per rollout (they sum to 1).
- **Activation** per decision: the shield fell back, or the *nominal* chunk
  (what the method executes without a shield: best g(a) for STL, Conf-Thresh
  and ShortStop; `candidates[0]` for MPC/CBF) was not accepted as proposed (a
  repaired chunk counts as rejected). A fallback re-decides after one step, so
  long stalls produce many activated decisions.
- **Fallback** per decision (per-decision flags are logged).
- **Recovery**: of the rollouts with any activation, the fraction with safe success.
- **Conservatism cost**: success lost against the unshielded rollout of the
  same scene/seed, on scenes where the unshielded one did not collide.
- **Latency added per decision**: `select` + g(a) for the shields that rank by
  g(a), `select` only for MPC/CBF; median (paper) with p95; a warm-up select
  runs before timing, then the heap is collected and frozen (`gc.freeze`):
  otherwise a full GC pass over the loaded model (~2.5 s, deterministic
  position) lands inside some method's select and sets its max.
- **Intervention precision is dropped**: its privileged rollout perturbs the
  episode (10/10 val scenes identical without it, 7/10 with it), and over the
  executed rows it is near 0 for every method.

## Final v2 (locked 2026-10-12, before the run)

Supersedes the v1 sections above where they differ (see
[`theory_audit.md`](theory_audit.md)); the tables above are rewritten after
the run.

- f-hat = pose model (DLS 0.03) for every model-based method and g(a);
  per-step `MODEL_ERROR` ε_1..ε_8 = 9.6 -> 36.7 mm (Eq. 1); `SAFETY_MARGIN`
  3 mm; fallback = joint hold anchored at the stall's first angles.
- **H = 4**: H sweep {4, 6, 8} on the 40 val scenes, rule fixed beforehand
  (highest mean safe success of ShortStop/MPC/CBF, tie -> 8): .800 / .792 / .775.
- **Rule H = r + d** (r = 4 executed rows per decision, d = inference delay
  in steps): Theorem 1 only covers rows inside the certified horizon, so H
  must cover every row executed before the next certificate. Without delay
  H = 4. The delay stress test reports both H = 4 (the final configuration,
  delayed rows uncertified) and H = 4 + d (rule), identically for
  ShortStop, MPC-Filter and CBF (CBF corrects 4 + d rows).
- Parameters: ShortStop eps 3 mm, Polyak, margin 1 mm, c = 5, K = 4 (not
  tuned); MPC-Filter none; CBF alpha 0.5 and Conf-Thresh 5 mm (v2 val
  tuning at H = 4); STL-Monitor eps 0.01 (user choice, ε_k = 0: no tube).
- Runs: `outputs/pipeline_v2/final.sh` -> `outputs/calvin_final/<baseline>_hold*`,
  then passive soundness logging on the first 300 scenes (ShortStop).
  Unshielded reuses the v1 1000 rollouts (rerun of 6 scenes identical to
  1e-6).
- Latency: warm-up, then `gc.collect()` + `gc.freeze()`.

## Delay stress test (registered 2026-10-14, before the run)

First 200 `final` scenes x 1 seed, inference delay d in {1, 2, 3} steps
(RTC-style `_Delay`). ShortStop, MPC-Filter, CBF in two modes with the final
parameters, nothing tuned: **naive** (H = 4, the final configuration; the
shield does not know the delay) and **aware** (`+delay_aware=true`: the
shield certifies the committed rows + rows d.. of each candidate over
H = 4 + d and never modifies the committed rows; rule H = r + d above).
Unshielded at each d as reference. Reported: safe success, violation,
paired tests aware vs naive and ShortStop vs each baseline. Fairness of
the comparison: [`fairness_audit.md`](fairness_audit.md).

## Component ablation and horizon sensitivity (registered 2026-10-14, before the run)

First 300 `final` scenes x 1 seed, H = 4 and every other parameter as in the
final; nothing tuned. Paired per scene with the rung above (same scenes and
policy seeds).

Component ladder (paper Table III / Fig. 3, each rung adds one component):

| rung | configuration | run |
|---|---|---|
| L0 | MDT, no shield | final `unshielded` rows |
| L1 | + certify (tube eps_k, margin, joint limits) + hold fallback, K = 1 | `reach_only` `+n_candidates=1` |
| L2 | + K = 4 candidates, first certified one in the policy's order | `reach_only` `+progress=false` |
| L3 | + g(a): best-g(a) certified candidate | `reach_only` |
| L4 | + counterexample-guided repair (= ShortStop) | final `shortstop_hold` rows |

`reach_only` certifies exactly like ShortStop (test
`test_reach_only_ablation_certifies_exactly_like_shortstop`). The paper's
"+STL-to-go" and "+CE search" rungs coincide with Reach-only on this spec:
for always-avoid, robustness-to-go is the tube's signed distance, so
`rho >= eps` is the tube-clearance test, and the capsule distance is exact,
so the counterexample adds nothing to certification (it only steers repair).

Horizon sensitivity (paper Fig. 4, Prop. 1): ShortStop at H in {4, 6, 8}
(H = 4: final rows). H < r = 4 leaves executed rows uncertified (Theorem 1)
and H > 8 exceeds the calibrated eps_1..eps_8 (chunk length 10), so the
sweep is limited to these three values. Expected from val (40 scenes): 0
violations at every H, success not increasing with H; reported as measured.

## Final evaluation (paper numbers)

1000 `final` scenes x 1 seed, K = 4, freeze, frozen `config.py` parameters,
`+precision=false`: `outputs/calvin_final/run_final.sh` (resumable; order
unshielded, ShortStop, MPC-Filter, CBF, Conf-Thresh, STL-Monitor; ~31 h, STL
alone ~17 h because it stalls). Val results that fixed these choices:
[`runs/2026-10-07_local_cluster_val/`](runs/2026-10-07_local_cluster_val/).

## Status

Done: scenes, g(a), TCP prediction, model_error, fallback (freeze), repair
(Polyak), local_cluster val tuning, final scenes, final v2, delay stress test,
component ablation and horizon sweep (runs/2026-10-16_delay_ablation).
Next: paper (docs/paper/shortstop).
Saved run outputs cited here are under [`runs/`](runs/).

## K probe in the final configuration (registered 2026-10-27, before the run)

The K = 1 / 4 / 8 probe that fixed K = 4 ran on 12 val scenes with the
superseded configuration (runs/2026-10-07_local_cluster_val). Repeated here
on all 40 `probe_local_cluster` val scenes with the final configuration
(hold fallback, per-step eps_k, 3 mm margin, H = 4, frozen config.py), for
ShortStop, K in {1, 4, 8}, 1 seed, `+precision=false`
(`shortstop_hold_kprobe_K<K>` in outputs/calvin_probe_local_cluster/).
Validation only: no test scene is run. K stays 4 whatever the outcome (every
reported run uses it); reported: safe success, violation, fallback, paired
discordant scenes vs K = 4, added latency by K. The paper text on K is
changed only after the authors review the result.

2026-10-27 22:3x: run stopped by the user at K = 8 after 17/40 scenes (K = 1
and K = 4 complete); the laptop was running other work during the run, so
its latencies are not valid. Outcomes do not depend on timing (synchronous
simulator), but latency must be re-measured on an idle machine: rerun the
three K values (or at least the latency) before reporting; K = 8 resumes
from scene 18 with the same command.
2026-10-28 01:43: the interrupted run is kept in
outputs/calvin_probe_local_cluster/_kprobe_busy_20261027/ (not reported);
K = 1, 4, 8 rerun from scratch on the idle machine.
2026-10-28 02:37, result (`python scripts/analyze_calvin_k_probe.py`):

| K | Safe success | Violation | Fallback | Added ms median / p95 / p99 / max | vs K = 4 |
|---|---|---|---|---|---|
| 1 | 80.0 % | 0.0 % | 27.6 % | 23 / 71 / 95 / 162 | +5.0 pts, p = 0.25 (2 vs 0 discordant) |
| 4 | 75.0 % | 0.0 % | 28.2 % | 60 / 255 / 303 / 330 | - |
| 8 | 75.0 % | 0.0 % | 35.6 % | 104 / 480 / 573 / 626 | +0.0 pts, p = 1.00 (2 vs 2) |

MDT sampling median 103 / 108 / 111 ms (one batch). The K = 4 rerun equals
`shortstop_hold_H4` in 40/40 scenes. No collision at any K; safe success
differs by at most two scenes (not significant); added latency grows about
linearly with K. K stays 4 (shared by every method; Conf-Thresh and
STL-Monitor need several candidates). Paper text on K unchanged (it cites
the earlier probe that fixed K); this probe is in the artifact
(`scripts/pipeline/k_probe.sh`, `scripts/analyze_calvin_k_probe.py`).
