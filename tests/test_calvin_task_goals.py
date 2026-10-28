"""g(a) and the shared task goals (shortstop.calvin.task_goals,
shortstop.calvin.progress): goals per task kind, phase switching, and the
TCP-based goal distance."""
import functools

import numpy as np
import pytest

from shortstop.calvin import static_targets
from shortstop.calvin import task_goals as tg
from shortstop.calvin.progress import calvin_progress_scores, subtask_goals
from shortstop.calvin.reach import CALVIN_ACTION_SCALE, propagate_arm_tube
from shortstop.calvin.robot_geometry import FLANGE_FRAME_INDEX, gripper_tip_position, panda_frames

Q_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
IDENTITY = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))


@pytest.fixture(autouse=True)
def _static_caches():
    """A drawer swept 0 -> 0.2 along -y, a button pressed 0 -> 0.02 down,
    no measured door contact points unless a test sets them."""
    static_targets.reset_cache_for_tests()
    static_targets.set_endpoint_cache_for_tests({
        "base__drawer": {"ll": {"min": [0.0, 0.0, 0.0], "max": [0.4, 0.1, 0.1]},
                         "ul": {"min": [0.0, -0.2, 0.0], "max": [0.4, -0.1, 0.1]}},
        "led": {"ll": {"min": [1.0, 1.0, 1.0], "max": [1.07, 1.07, 1.03]},
                "ul": {"min": [1.0, 1.0, 0.98], "max": [1.07, 1.07, 1.01]}},
    })
    static_targets.set_joint_range_cache_for_tests({"base__drawer": {"ll": 0.0, "ul": 0.2},
                                                    "led": {"ll": 0.0, "ul": 0.02}})
    static_targets.set_door_contact_cache_for_tests({})
    yield
    static_targets.reset_cache_for_tests()


def _scene(red=(0.05, -0.12, 0.46), blue=(0.07, 0.092, 0.46), pink=(0.23, -0.12, 0.46), drawer=0.0, button=0.0):
    return {
        "movable_objects": {"block_red": {"current_pos": red}, "block_blue": {"current_pos": blue},
                            "block_pink": {"current_pos": pink}},
        "doors": {"base__drawer": {"current_state": drawer}, "base__slide": {"current_state": 0.0}},
        "buttons": {"base__button": {"joint_state": button}},
        "switches": {"base__switch": {"joint_state": 0.0}},
        "fixed_objects": {"table": {}},
    }


def _goals(task, spec, scene, tip=(0.0, 0.0, 0.0)):
    return tg.task_goals(task, spec, scene, np.asarray(tip), *IDENTITY)


# --- goals per task kind ---------------------------------------------------

def test_push_goal_is_the_block_then_the_block_pushed():
    contact, completion = _goals("push_red_block_right", ("push_object", "block_red", 0.1, 0), _scene())
    assert np.allclose(contact.box_min, [0.05, -0.12, 0.46]) and np.allclose(contact.box_min, contact.box_max)
    assert np.allclose(completion.box_min, [0.15, -0.12, 0.46])


def test_lift_and_rotate_goals():
    _, lift = _goals("lift_red_block_table", ("lift_object", "block_red", 0.05, "table", "base_link"), _scene())
    assert np.allclose(lift.box_min, [0.05, -0.12, 0.46 + 0.05 + tg.LIFT_EXTRA])
    contact, rotate = _goals("rotate_red_block_left", ("rotate_object", "block_red", 60), _scene())
    assert rotate.box_min is None and np.isclose(rotate.wrist_turn, -np.radians(60))


def test_door_goal_is_the_measured_contact_point_moving_with_the_joint():
    static_targets.set_door_contact_cache_for_tests({"open_drawer": [0.2, 0.05, 0.04]})
    contact, completion = _goals("open_drawer", ("move_door_rel", "base__drawer", 0.12), _scene(drawer=0.0))
    assert np.allclose(contact.box_min, [0.2, 0.05, 0.04]) and np.allclose(contact.box_min, contact.box_max)
    assert np.allclose(completion.box_min, [0.2, -0.07, 0.04])  # box_min moved by -0.12 in y


def test_door_goal_falls_back_to_the_link_box_without_a_measured_contact():
    contact, _ = _goals("open_drawer", ("move_door_rel", "base__drawer", 0.12), _scene())
    assert np.allclose(contact.box_min, [0.0, 0.0, 0.0]) and np.allclose(contact.box_max, [0.4, 0.1, 0.1])


def test_button_goal_is_the_button_then_the_button_pressed():
    contact, completion = _goals("turn_on_led", ("toggle_light", "led", 0, 1), _scene())
    assert np.allclose(contact.box_min, [1.0, 1.0, 1.0])
    assert np.allclose(completion.box_min, [1.0, 1.0, 0.98])


def test_push_into_drawer_picks_the_table_block_nearest_the_start_tip():
    spec = ("push_object_into", ["block_red", "block_blue", "block_pink"], "table", "base_link", "table", "drawer_link")
    # blue is in the slider (y > 0) -- never picked, even when nearest.
    contact, completion = _goals("push_into_drawer", spec, _scene(), tip=(0.07, 0.09, 0.46))
    assert contact.meta["obj"] == "block_red"
    contact, _ = _goals("push_into_drawer", spec, _scene(), tip=(0.23, -0.12, 0.6))
    assert contact.meta["obj"] == "block_pink"
    assert np.allclose(completion.box_max, [0.4, 0.1, 0.1])  # the drawer's box


def test_goals_are_in_the_robot_base_frame():
    contact, _ = tg.task_goals("push_red_block_right", ("push_object", "block_red", 0.1, 0), _scene(),
                               np.zeros(3), (-0.34, -0.46, 0.24), (0.0, 0.0, 0.0, 1.0))
    assert np.allclose(contact.box_min, np.array([0.05, -0.12, 0.46]) - np.array([-0.34, -0.46, 0.24]))


# --- phase switching -------------------------------------------------------

def test_block_task_switches_to_completion_once_the_block_moves_and_stays():
    goals = tg.SubtaskGoals("push_red_block_right", ("push_object", "block_red", 0.1, 0), _scene(), np.zeros(3), *IDENTITY)
    assert np.allclose(goals.target(_scene(red=(0.055, -0.12, 0.46)))[0], [0.05, -0.12, 0.46])  # 5 mm: still phase 1
    assert np.allclose(goals.target(_scene(red=(0.07, -0.12, 0.46)))[0], [0.15, -0.12, 0.46])   # 2 cm: phase 2
    assert np.allclose(goals.target(_scene())[0], [0.15, -0.12, 0.46])                           # latched


def test_joint_task_switches_at_a_tenth_of_its_travel():
    goals = tg.SubtaskGoals("turn_on_led", ("toggle_light", "led", 0, 1), _scene(), np.zeros(3), *IDENTITY)
    assert np.allclose(goals.target(_scene(button=0.001))[0], [1.0, 1.0, 1.0])     # 1 mm of 20: phase 1
    assert np.allclose(goals.target(_scene(button=0.003))[0], [1.0, 1.0, 0.98])    # 3 mm: phase 2


def test_rotate_keeps_its_contact_goal_after_the_block_moves():
    goals = tg.SubtaskGoals("rotate_red_block_left", ("rotate_object", "block_red", 60), _scene(), np.zeros(3), *IDENTITY)
    assert np.allclose(goals.target(_scene(red=(0.1, -0.12, 0.46)))[0], [0.05, -0.12, 0.46])


def test_subtask_goals_reads_the_task_oracle_and_is_none_without_a_goal():
    def push_object(*_a, **_k):
        return None

    def stack_objects(*_a, **_k):
        return None

    oracle = type("Oracle", (), {})()
    oracle.tasks = {"push_red_block_right": functools.partial(push_object, "block_red", 0.1, 0),
                    "stack_block": functools.partial(stack_objects)}
    goals = subtask_goals(oracle, "push_red_block_right", {"scene_info": _scene()}, Q_HOME)
    assert goals is not None and np.allclose(goals.target(_scene())[0], [0.05, -0.12, 0.46])
    assert subtask_goals(oracle, "stack_block", {"scene_info": _scene()}, Q_HOME) is None
    assert subtask_goals(oracle, "missing", {"scene_info": _scene()}, Q_HOME) is None


# --- the score -------------------------------------------------------------

def _x_chunk(dx_m, rows=4):
    chunk = np.zeros((rows, 7))
    chunk[:, 0] = dx_m / CALVIN_ACTION_SCALE
    return chunk


def test_score_prefers_the_candidate_whose_tcp_ends_closer_and_is_neutral_without_goal():
    tip = gripper_tip_position(Q_HOME)
    goal = (tip + [0.05, 0, 0], tip + [0.05, 0, 0])
    toward, away = _x_chunk(0.01), _x_chunk(-0.01)
    scores = calvin_progress_scores(Q_HOME, [toward, away], goal)
    assert scores[0] > scores[1]
    assert calvin_progress_scores(Q_HOME, [toward, away], None) == [0.0, 0.0]


def test_score_measures_the_tcp_not_the_flange():
    chunk = _x_chunk(0.01)
    tube = propagate_arm_tube(Q_HOME, chunk, w_bar=0.0, model_error=0.0)
    tcp, flange = tube[-1]["fingertip"].b, tube[-1][FLANGE_FRAME_INDEX].center()
    assert np.linalg.norm(tcp - flange) > 0.1
    assert calvin_progress_scores(Q_HOME, [chunk], (tcp, tcp))[0] == pytest.approx(0.0, abs=1e-12)
    assert calvin_progress_scores(Q_HOME, [chunk], (flange, flange))[0] < -0.1


def test_score_uses_the_whole_chunk_it_is_given():
    tip = gripper_tip_position(Q_HOME)
    goal = (tip + [0.08, 0, 0], tip + [0.08, 0, 0])
    long_chunk = _x_chunk(0.01, rows=8)   # 1 cm per row: 4 rows reach +4 cm, 8 rows +8 cm
    prefix, full = calvin_progress_scores(Q_HOME, [long_chunk[:4], long_chunk], goal)
    assert full > prefix


def test_predicted_tcp_follows_the_commanded_displacement():
    """Actions command the TCP, so a pure x command moves the predicted TCP
    along x by (about) that much -- the flange need not."""
    tube = propagate_arm_tube(Q_HOME, _x_chunk(0.02, rows=1), w_bar=0.0, model_error=0.0)
    moved = tube[-1]["fingertip"].b - gripper_tip_position(Q_HOME)
    assert np.allclose(moved, [0.02, 0.0, 0.0], atol=2e-4)
    assert np.linalg.norm(panda_frames(Q_HOME)[FLANGE_FRAME_INDEX]) > 0  # (sanity: frames still available)
