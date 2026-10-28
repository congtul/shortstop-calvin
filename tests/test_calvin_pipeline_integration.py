"""End-to-end structural check of the P-R-C-S loop on synthetic data:
MockMDTPolicyClient.propose() -> ArmRepairShield.select(), no CALVIN/MDT
session -- checks the wiring, not the numbers.
"""
import numpy as np

from shortstop.calvin.reach import propagate_arm_tube
from shortstop.calvin.shield import ArmConfThreshShield, ArmRepairShield, ArmSTLMonitorShield
from shortstop.env import Obstacle
from shortstop.calvin.mdt_policy_client import MockMDTPolicyClient
from shortstop.calvin.robot_geometry import FLANGE_FRAME_INDEX, N_JOINTS

# A real, well-within-JOINT_LIMITS Franka "ready" pose -- NOT np.zeros: see
# tests/test_arm_shield.py's own Q_HOME for why q=0 is itself physically
# invalid for joint 4 alone, which matters now that select() enforces
# JOINT_LIMITS.
Q_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])


def test_full_propose_reach_certify_repair_loop_on_synthetic_candidates():
    rng = np.random.default_rng(0)
    q = Q_HOME
    policy = MockMDTPolicyClient(horizon=10, action_dim=7, n_candidates=6, noise_std=0.05, rng=rng)

    candidates = policy.propose(observation={})  # Propose
    assert len(candidates) == 6

    endpoints = []
    for c in candidates:
        tube = propagate_arm_tube(q, c, w_bar=0.0, model_error=0.0)  # Reach
        endpoints.append(tube[-1][FLANGE_FRAME_INDEX].center())
    target = max(endpoints, key=lambda p: p[0])
    obstacle = Obstacle(center=target, radius=0.05)

    shield = ArmRepairShield(
        obstacles=[obstacle], w_bar=0.01, model_error=0.02, epsilon=0.02,
        trust_region=0.3, step_size=0.05, max_repair_iters=3,
    )
    scores = list(range(len(candidates)))
    action, info = shield.select(q, candidates, scores)  # Certify -> Select/Repair

    assert action.shape == (10, 7)
    assert len(info["admissible_mask"]) == len(candidates)
    assert (not all(info["admissible_mask"])) or info["repair_attempted"] or info["fallback"]


def test_full_propose_select_loop_with_conf_thresh_shield_on_synthetic_candidates():
    """Conf-Thresh's own P-R-C-S variant: no Reach/obstacle knowledge at
    all, disagreement-filter -> Select instead of tube-vs-obstacle
    Certify -> Repair (see shortstop.calvin.shield.ArmConfThreshShield)."""
    rng = np.random.default_rng(0)
    q = Q_HOME
    policy = MockMDTPolicyClient(horizon=10, action_dim=7, n_candidates=8, noise_std=0.05, rng=rng)

    candidates = policy.propose(observation={})  # Propose
    assert len(candidates) == 8

    shield = ArmConfThreshShield(disagreement_threshold=10.0, horizon_steps=10)  # generous: exercises wiring, not exact filtering
    scores = list(range(len(candidates)))
    action, info = shield.select(q, candidates, scores)  # disagreement-filter -> Select

    assert action.shape == (10, 7)
    assert len(info["admissible_mask"]) == len(candidates)
    assert len(info["disagreement"]) == len(candidates)
    assert not info["fallback"]  # threshold generous enough that at least one candidate is admissible


def test_full_propose_select_loop_with_stl_monitor_shield_on_synthetic_candidates():
    """STL-Monitor's own P-R-C-S variant: nominal STL robustness (no
    reachtube, no repair) filter -> Select instead of Conf-Thresh's
    disagreement filter or ShortStop's certified tube+repair (see
    shortstop.calvin.shield.ArmSTLMonitorShield)."""
    rng = np.random.default_rng(0)
    q = Q_HOME
    policy = MockMDTPolicyClient(horizon=10, action_dim=7, n_candidates=8, noise_std=0.05, rng=rng)

    candidates = policy.propose(observation={})  # Propose
    assert len(candidates) == 8

    # obstacle far from every candidate's path -- exercises wiring, not exact filtering
    obstacle = Obstacle(center=np.array([10.0, 10.0, 10.0]), radius=0.05)
    shield = ArmSTLMonitorShield(obstacles=[obstacle], epsilon=0.0)
    scores = list(range(len(candidates)))
    action, info = shield.select(q, candidates, scores)  # nominal-STL-filter -> Select

    assert action.shape == (10, 7)
    assert len(info["admissible_mask"]) == len(candidates)
    assert not info["fallback"]  # obstacle far away -> every candidate admissible
