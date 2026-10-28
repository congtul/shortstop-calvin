"""Constants shared by the CALVIN harness, shields, scene builder and run
scripts."""
import os

# Candidate chunks per decision, same for every baseline (MPC-Filter/CBF only
# correct candidates[0]). K=4: the K probe (1/4/8, 40 val scenes) showed no
# gain from more MDT samples (mm spread), and ShortStop's latency scales with K.
N_CANDIDATES = 4

# MDT's action-chunk length (act_window_size). Fixed by the checkpoint.
CHUNK_LENGTH = 10

# Rows of the selected chunk executed before proposing again: one decision
# = one policy call = one shield call, for every baseline.
REPLAN_STEPS = 4

# Rows certified per decision (the paper's H): 4 = REPLAN_STEPS, chosen on
# val and used for every reported run. The SHORTSTOP_HORIZON environment
# variable overrides it (H sweep: 6, 8; delay-aware runs add d themselves).
HORIZON_STEPS = int(os.environ.get("SHORTSTOP_HORIZON", 4))

assert 0 < REPLAN_STEPS <= HORIZON_STEPS <= CHUNK_LENGTH

# Obstacle sphere radius (m) of the frozen scenes.
OBSTACLE_RADIUS = 0.05

# Shields' model-error bound (m) per horizon step k = 1..8: p99 x 1.25 of the
# k-step position error of the pose f-hat over every arm frame and the TCP,
# ~2600 decisions executing 8 rows open loop
# (scripts/calibrate_arm_model_error.py, run_20261012_120501). The error grows
# with k, so the tube does (paper Eq. 1, Assumption 1). The env is
# deterministic, so w_bar = 0.
MODEL_ERROR = (0.00958, 0.01471, 0.0185, 0.02239, 0.02649, 0.02941, 0.03384, 0.03666)
# f-hat follows the rotation columns too (reach.step_joint_config with the TCP
# orientation, which the harness passes to every shield and to g(a)).
POSE_MODEL = True
# Certified clearance (m) every tube-based shield keeps (ShortStop's epsilon,
# MPC-Filter/CBF constraint margin): covers the arm's drift under the
# joint-hold fallback (calibrate_arm_model_error.py), so X_s = {dist >= eps}
# stays invariant while the fallback runs (Assumption 3).
SAFETY_MARGIN = 0.003  # joint-hold drift: p99 1.1 mm over 8 held steps (max 1.4 mm)
# Joint-space error margin (rad) on the joint limits the shields check. The
# certified spec is obstacle avoidance only: the pose f-hat's joint error is
# large (redundant arm; p99 x 1.25 0.05 -> 0.44 rad over 8 steps), so joint
# limits are a nominal filter (0), enforced on a robot by its controller.
JOINT_ERROR = 0.0
W_BAR = 0.0

# Scene builder: an obstacle must stay radius + fingertip radius + this far
# from every task goal. Fixed at the value the frozen val/test scenes were
# built with (independent of MODEL_ERROR).
SCENE_TARGET_MARGIN = 0.02

# Distance (m) the retreat fallback (an ablation; freeze is the default) moves
# the arm away from the obstacle: one maximal policy action (|a| = 1).
# 2.5 cm was no better on val (84.7 % vs 84.0 % safe success, 150 rollouts).
FALLBACK_RETREAT_STEP = 0.02

# Baseline hyperparameters, tuned on the 40 val scenes (probe_local_cluster)
# with the final configuration (pre_final.sh: H = 4, hold fallback) and frozen
# for every reported run (outputs/calvin_tuning_probe_local_cluster/*_v2_chosen.json).
CONF_THRESH_THRESHOLD = 0.005
# 0.01 by choice (first val protocol, kept): 0 had higher safe success (0.45 vs 0.325) but 5x the violations
# (0.125 vs 0.025); a safety filter trades success for safety.
STL_MONITOR_EPSILON = 0.01
CBF_ALPHA = 0.5
# ShortStop has no tuned parameter: epsilon = SAFETY_MARGIN like the MPC/CBF constraints,
# repair_margin = SQP_MARGIN (shield.py), at most 5 re-linearizations per candidate.
SHORTSTOP_PARAMS = {"epsilon": SAFETY_MARGIN, "repair_mode": "polyak", "repair_margin": 0.001, "max_linearized_iters": 5}
