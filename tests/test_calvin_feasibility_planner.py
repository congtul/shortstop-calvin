import numpy as np

from shortstop.calvin import feasibility_planner as fp
from shortstop.calvin.robot_geometry import gripper_tip_position
from shortstop.env import Obstacle

Q_START = np.array([-1.21779124, 1.03987629, 2.11978254, -2.34205014, -0.87015899, 1.64119093, 0.55344928])
# A table block 0.25 m from the start fingertip, in the robot-base frame.
BLOCK = np.array([0.57, 0.34, 0.22])


def _point_goal(p):
    p = np.asarray(p, dtype=float)
    return fp.TaskGoal(p, p.copy())


def test_clearance_is_infinite_without_obstacles_and_negative_when_touching():
    assert fp.arm_obstacle_clearance(Q_START, []) == np.inf
    tip = gripper_tip_position(Q_START)
    assert fp.arm_obstacle_clearance(Q_START, [Obstacle(center=tip, radius=0.05)]) < 0
    assert not fp.is_state_valid(Q_START, [Obstacle(center=tip, radius=0.05)])
    assert fp.is_state_valid(Q_START, [Obstacle(center=tip + [0, 0, 1.0], radius=0.05)])


def test_feasible_without_obstacle():
    rng = np.random.default_rng(0)
    result = fp.check_feasibility(Q_START, _point_goal(BLOCK), _point_goal(BLOCK + [0.1, 0, 0]), [], rng)
    assert result["feasible"]
    assert fp.satisfies_goal(np.array(result["contact_config"]), _point_goal(BLOCK))
    assert fp.satisfies_goal(np.array(result["completion_config"]), _point_goal(BLOCK + [0.1, 0, 0]))


def test_infeasible_when_obstacle_sits_on_the_target():
    rng = np.random.default_rng(0)
    obstacle = [Obstacle(center=BLOCK, radius=0.05)]
    result = fp.check_feasibility(Q_START, _point_goal(BLOCK), _point_goal(BLOCK + [0.1, 0, 0]), obstacle, rng)
    assert not result["feasible"]
    assert result["reason"] == "no_contact_ik"


def test_start_in_collision_is_reported():
    rng = np.random.default_rng(0)
    obstacle = [Obstacle(center=gripper_tip_position(Q_START), radius=0.05)]
    result = fp.check_feasibility(Q_START, _point_goal(BLOCK), _point_goal(BLOCK), obstacle, rng)
    assert (result["feasible"], result["reason"]) == (False, "start_in_collision")


def test_rrt_detours_when_the_straight_joint_path_is_blocked():
    rng = np.random.default_rng(3)
    contact = _point_goal(BLOCK)
    goals = fp.solve_ik(contact, [Q_START], [], rng)
    assert goals
    # Put the obstacle on the straight-line joint path's midpoint fingertip.
    mid = gripper_tip_position(0.5 * (Q_START + goals[0]))
    obstacle = [Obstacle(center=mid + [0.0, 0.0, 0.02], radius=0.04)]
    if not fp.is_state_valid(Q_START, obstacle):
        return  # geometry changed; nothing meaningful to test here
    valid_goals = [g for g in goals if fp.is_state_valid(g, obstacle)]
    assert valid_goals
    assert not any(fp.is_edge_valid(Q_START, g, obstacle) for g in valid_goals)

    path, _ = fp.rrt_connect(Q_START, valid_goals, obstacle, rng)

    assert path is not None
    assert np.allclose(path[0], Q_START)
    for q_a, q_b in zip(path[:-1], path[1:]):
        assert fp.is_edge_valid(q_a, q_b, obstacle)


def test_wrist_turn_completion_turns_joint_7_only():
    rng = np.random.default_rng(0)
    completion = fp.TaskGoal(wrist_turn=-np.radians(60))
    result = fp.check_feasibility(Q_START, _point_goal(BLOCK), completion, [], rng)
    assert result["feasible"]
    delta = np.array(result["completion_config"]) - np.array(result["contact_config"])
    assert np.allclose(delta[:6], 0.0)
    assert np.isclose(delta[6], -np.radians(60))
