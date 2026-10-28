# ShortStop on CALVIN: protocol and configuration

This is the configuration every number in the paper was produced with. The
chronological record of how it was reached (pivots, superseded values,
decisions and the reasons for them) is [`lab_notebook.md`](lab_notebook.md);
where the two disagree, this file is the one that matches the runs.
How to rerun everything: [`../../REPRODUCE.md`](../../REPRODUCE.md).

## Benchmark and policy

- CALVIN, environment D (`calvin_scene_D`, PyBullet), with the public MDT
  checkpoint `mdtv-1-abcd` (trained on the ABCD→D split, file
  `avg_seq_len=4.64.ckpt`), 10 DDIM sampling steps, chunks of
  `CHUNK_LENGTH = 10` relative TCP actions. The policy is frozen.
- One language-conditioned task per rollout from a fresh CALVIN initial
  state, `ep_len = 360` steps. 27 tasks: every task that can start from a
  fresh state, stacking excluded ([`task_set.md`](task_set.md)).
- A rollout ends at task success (CALVIN's oracle), at the first collision
  of any arm link or finger capsule with an obstacle, or at the step limit:
  **safe success**, **violation** or **timeout**.

## Scenes

Frozen in [`shortstop/calvin/data/scenes/`](../../shortstop/calvin/data/scenes/)
(task, initial state, obstacles, policy seed), built offline by
`scripts/build_calvin_scene_manifest.py` with the `local_cluster` rules
([`scene_build_rules.md`](scene_build_rules.md)):

- 1-3 spheres, radius 3-7 cm, at least 2 cm apart, within 20 cm of an
  anchor near the task's contact goal or on the arm's reference path; not
  rendered, so the policy cannot see them.
- Kept only if the start pose clears every sphere by 5 cm, the spheres keep
  out of the task targets, 20-80 % of reference unshielded rollouts touch
  them, and an RRT-Connect planner finds a safe start-contact-completion
  path with a 2 cm margin.

| Split | File | Scenes | Use |
|---|---|---|---|
| val | `probe_local_cluster_scenes.json` | 40 | H selection, CBF / Conf-Thresh tuning |
| test | `final_scenes.json` | 1000 (97 initial states, disjoint from val) | every reported result |

The other scene files (`val_`, `test_`, `probe_*`) are the earlier pilot
sets, kept so the run records that cite them stay reproducible.

## Decision loop

Every `REPLAN_STEPS = r = 4` env steps (`shortstop/calvin/experiment.py`):

1. **Propose**: MDT samples `N_CANDIDATES = K = 4` chunks in one batched
   call (`mdt_policy_client.py`).
2. **Reach**: the first `HORIZON_STEPS = H = 4` rows of each chunk are
   propagated through f-hat (pose model: TCP position + orientation targets
   to joint steps by a damped least-squares Jacobian inverse, λ = 0.03, then
   forward kinematics) into capsules of the whole arm and fingers, inflated
   at step k by `MODEL_ERROR[k-1]` (ε_1..ε_8 = 9.6, 14.7, 18.5, 22.4, 26.5,
   29.4, 33.8, 36.7 mm; `reach.py`).
3. **Certify**: STL robustness-to-go of "always avoid" over the tube = the
   smallest capsule-sphere distance over k = 0..H; certified if at least
   `SAFETY_MARGIN = 3 mm` (`shield.py`).
4. **Select / repair / fall back**: the certified candidate with the best
   task-progress score g(a) (`progress.py`, `task_goals.py`); if none
   certifies, Polyak repair of the candidates in g(a) order (closed form
   along the worst counterexample, re-linearized, at most 5 rounds, 1 mm
   margin, translation columns, rows capped at 10 cm); if repair fails, the
   **joint-hold fallback** holds the joint angles where the stall began for
   one step and the loop proposes again.

Collisions are judged on the true arm (`experiment._clearance`: 8 link
capsules plus both finger capsules at the current gripper width).

## Parameters (`shortstop/calvin/config.py`)

| Parameter | Value | How it was set |
|---|---|---|
| `N_CANDIDATES` (K) | 4 | design; K = 1/4/8 on the 40 val scenes: no collision, safe success within 2 scenes, latency grows with K (`scripts/analyze_calvin_k_probe.py`) |
| `CHUNK_LENGTH` / `REPLAN_STEPS` (r) | 10 / 4 | checkpoint / design |
| `HORIZON_STEPS` (H) | 4 | val sweep {4, 6, 8}, rule fixed beforehand (`scripts/pipeline/choose_h.py`) |
| `MODEL_ERROR` (ε_k) | 9.6 → 36.7 mm | p99 × 1.25 of the k-step error, 50 obstacle-free rollouts, ~2600 decisions (`calibrate_arm_model_error.py`, run `20261012_120501`) |
| `W_BAR` | 0 | the simulator is deterministic ([`env_verification.md`](env_verification.md)) |
| `SAFETY_MARGIN` (ε) | 3 mm | covers the joint-hold drift (≤ 0.25 mm over 300 held steps) |
| `SHORTSTOP_PARAMS` | ε = 3 mm, Polyak, 1 mm, 5 rounds | not tuned |
| `CBF_ALPHA` | 0.5 | val tuning at H = 4 |
| `CONF_THRESH_THRESHOLD` | 5 mm | val tuning at H = 4 |
| `STL_MONITOR_EPSILON` | 1 cm (ε_k = 0, no tube) | chosen on the first val protocol, kept |
| `JOINT_ERROR` | 0 | joint limits are a nominal filter; the certified spec is obstacle avoidance |

## Baselines

All model-based baselines share f-hat, the capsule geometry, the ε_k
tightening, the 3 mm margin and the hold fallback
([`fairness_audit.md`](fairness_audit.md)).

| Method | Class (`shield.py`) | Ranks by g(a) | What it does |
|---|---|---|---|
| Unshielded | none | no | executes the policy's first sample |
| Conf-Thresh | `ArmConfThreshShield` | yes | rejects candidates whose predicted TCP endpoint is > 5 mm from the candidates' centroid; no obstacle knowledge |
| STL-Monitor | `ArmSTLMonitorShield` | yes | nominal robustness of the f-hat rollout ≥ 1 cm, no tube, no repair |
| MPC-Filter | `ArmMPCFilterShield` | no | H-step problem, minimum deviation from the policy's chunk, tightened collision and joint-limit constraints, SQP (≤ 5 QPs) |
| CBF | `ArmCBFShield` | no | per-row discrete-time CBF QP, h(q_{k+1}) ≥ (1 - α) h(q_k), joint limits |
| Reach-only (ablation) | `ArmReachOnlyShield` | yes / `+progress=false` | certifies exactly like ShortStop, no repair |
| ShortStop | `ArmRepairShield` | yes | as above |

Infeasibility or no certified chunk triggers the fallback for every method.

## Metrics (`baseline_runner.rollout_metrics`)

- Safe success, violation, timeout per rollout (they sum to 1).
- Activation per decision: a fallback ran, or the chunk the method would run
  unshielded (best g(a) for the g(a) methods, the first sample for MPC/CBF)
  was not accepted as proposed. Fallback per decision.
- Conservatism cost: success lost against the unshielded rollout of the
  same scene, on the scenes where the unshielded rollout does not collide.
- Added latency per decision: selection, repair and solvers, plus g(a) for
  the methods that rank by it; one warm-up decision first, then
  `gc.collect()` + `gc.freeze()`.
- 95 % bootstrap CIs over scenes (10^4 resamples); paired bootstrap tests
  over the same scenes.

## Experiments

Each was registered in [`lab_notebook.md`](lab_notebook.md) before it ran.

| Paper | Experiment | Stage script | Rollouts (`outputs/calvin_final/`) | Run record |
|---|---|---|---|---|
| Setup (H, tuning) | H sweep and tuning, 40 val scenes | `pipeline/pre_final.sh` | `../calvin_probe_local_cluster/`, `../calvin_tuning_probe_local_cluster/` | [`lab_notebook.md`](lab_notebook.md), "Final v2" |
| Setup (K) | K probe, K = 1, 4, 8, 40 val scenes | `pipeline/k_probe.sh` | `../calvin_probe_local_cluster/shortstop_hold_kprobe_K<K>.jsonl` | [`lab_notebook.md`](lab_notebook.md), "K probe in the final configuration" |
| Table 1, Fig. 2 | main comparison, 1000 scenes | `pipeline/final.sh` | `<method>_hold.jsonl`, `unshielded.jsonl` | [`runs/2026-10-14_final_v2/`](runs/2026-10-14_final_v2/) |
| Sect. 6.2 | soundness log, first 300 scenes | `pipeline/final.sh` | `shortstop_hold_soundness_smoke.jsonl` | [`runs/2026-10-14_final_v2/`](runs/2026-10-14_final_v2/) |
| Table 3 | inference delay d = 1..3, first 200 scenes | `pipeline/delay.sh` | `*_delay<d>_*_smoke.jsonl` | [`runs/2026-10-16_delay_ablation/`](runs/2026-10-16_delay_ablation/) |
| Table 2 | component ablation, first 300 scenes | `pipeline/ablation.sh full` | `reach_only_hold_abl_L<i>_smoke.jsonl` | [`runs/2026-10-16_delay_ablation/`](runs/2026-10-16_delay_ablation/) |
| Sect. 7 | horizon H = 4, 6, 8, first 300 scenes | `pipeline/ablation.sh full` | `shortstop_hold_H<H>_smoke.jsonl` | [`runs/2026-10-16_delay_ablation/`](runs/2026-10-16_delay_ablation/) |

The `_smoke` suffix only marks a run limited to the first N scenes
(`+limit=N`). Every rollout is seeded by its scene (`policy_seed`), so a
rollout is the same whether it runs alone, in a subset or after a restart.

Other documents: [`theory_audit.md`](theory_audit.md) (every paper
assumption and equation against the code), [`fairness_audit.md`](fairness_audit.md),
[`env_verification.md`](env_verification.md), [`runs/`](runs/) (the earlier
studies the notebook cites).
