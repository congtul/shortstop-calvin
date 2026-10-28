import json

import numpy as np
import pytest

from shortstop.calvin import scene_manifest as sm
from shortstop.calvin import static_targets
from shortstop.calvin.robot_geometry import gripper_tip_position, to_world_frame
from shortstop.env import Obstacle


@pytest.fixture(autouse=True)
def _isolated_static_targets():
    static_targets.reset_cache_for_tests()
    static_targets.set_door_contact_cache_for_tests({})  # never the real measured file
    yield
    static_targets.reset_cache_for_tests()


def _scene_obs():
    obs = np.zeros(24)
    obs[6:9] = [0.05, -0.12, 0.46]    # red on table
    obs[12:15] = [0.23, -0.12, 0.46]  # blue on table
    obs[18:21] = [0.07, 0.092, 0.46]  # pink in slider
    return obs


def _local(world):
    return world - np.asarray(sm.BASE_POSITION)


def test_start_pose_matches_calvins_reset_tcp():
    tip_world = to_world_frame(gripper_tip_position(sm.START_JOINT_ANGLES), sm.BASE_POSITION, sm.BASE_ORIENTATION)
    assert np.allclose(tip_world, [0.02586889, -0.2313129, 0.5712808], atol=1e-5)


def test_push_goals_follow_the_block_and_the_push_vector():
    contact, completion = sm.task_goals("push_red_block_right", ("push_object", "block_red", 0.1, 0), _scene_obs())
    assert np.allclose(contact.box_min, _local(np.array([0.05, -0.12, 0.46])))
    assert np.allclose(completion.box_min, _local(np.array([0.15, -0.12, 0.46])))


def test_lift_and_rotate_goals():
    _, lift = sm.task_goals("lift_blue_block_table", ("lift_object", "block_blue", 0.05, "table", "base_link"), _scene_obs())
    assert np.allclose(lift.box_min, _local(np.array([0.23, -0.12, 0.46 + 0.05 + sm.LIFT_EXTRA])))
    _, rotate = sm.task_goals("rotate_red_block_left", ("rotate_object", "block_red", 60), _scene_obs())
    assert rotate.box_min is None and np.isclose(rotate.wrist_turn, -np.radians(60))


def test_door_goals_use_the_box_at_the_current_and_target_joint_value():
    static_targets.set_endpoint_cache_for_tests({"base__drawer": {
        "ll": {"min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]},
        "ul": {"min": [0.0, -0.22, 0.0], "max": [0.0, -0.22, 0.0]},
    }})
    static_targets.set_joint_range_cache_for_tests({"base__drawer": {"ll": 0.0, "ul": 0.22}})
    contact, completion = sm.task_goals("open_drawer", ("move_door_rel", "base__drawer", 0.12), _scene_obs())
    assert np.allclose(contact.box_min, _local(np.zeros(3)))
    assert np.allclose(completion.box_min, _local(np.array([0.0, -0.12, 0.0])))


def test_push_into_drawer_picks_a_table_block_never_a_slider_one():
    static_targets.set_endpoint_cache_for_tests({"base__drawer": {
        "ll": {"min": [0, 0, 0], "max": [1, 1, 1]}, "ul": {"min": [0, 0, 0], "max": [1, 1, 1]}}})
    static_targets.set_joint_range_cache_for_tests({"base__drawer": {"ll": 0.0, "ul": 0.22}})
    obs = _scene_obs()
    contact, _ = sm.task_goals(
        "push_into_drawer",
        ("push_object_into", ["block_pink", "block_red", "block_blue"], "table", "base_link", "table", "drawer_link"),
        obs,
    )
    assert contact.meta["obj"] in ("block_red", "block_blue")


def test_missing_static_cache_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(static_targets, "CACHE_PATH", tmp_path / "does_not_exist.json")
    with pytest.raises(ValueError):
        sm.task_goals("open_drawer", ("move_door_rel", "base__drawer", 0.12), _scene_obs())


def test_partition_is_disjoint_and_complete():
    val, test = sm.partition_states(list(range(192)), 0.5, rng_seed=0)
    assert not set(val) & set(test)
    assert sorted(val + test) == list(range(192))
    assert len(val) == 96


def test_allocation_is_even():
    names = [f"t{i}" for i in range(26)]
    allocation = sm.allocate_trials(names, 100, rng_seed=0)
    counts = [allocation.count(n) for n in names]
    assert len(allocation) == 100 and min(counts) == 3 and max(counts) == 4


def test_rollout_hit_rate_counts_rollouts_not_steps():
    frames = np.zeros((4, 9, 3))
    frames[:, :, 0] = np.arange(9) * 0.1        # arm along +x
    frames[2:, :, 2] = 1.0                       # rollout 1 is lifted 1 m
    rollout_id = np.array([0, 0, 1, 1])
    hit = Obstacle(center=np.array([0.35, 0.0, 0.0]), radius=0.01)
    assert sm.rollout_hit_rate(frames, rollout_id, hit) == 0.5


def test_load_scenes_rebuilds_obstacles_and_rechecks_scene_obs(tmp_path):
    scene = {"scene_id": "pilot_0000", "task_name": "open_drawer", "initial_state": {"drawer": "closed"},
             "scene_obs": [0.0] * 24, "obstacle_center_local": [0.1, 0.2, 0.3], "obstacle_radius": 0.05}
    path = tmp_path / "scenes.json"
    path.write_text(json.dumps({"version": 1, "scenes": [scene]}))

    _, scenes = sm.load_scenes(path, lambda state: (None, np.zeros(24)))
    assert scenes[0]["eval_sequence"] == ("open_drawer",)
    assert np.allclose(scenes[0]["obstacles"][0].center, [0.1, 0.2, 0.3])

    with pytest.raises(ValueError):
        sm.load_scenes(path, lambda state: (None, np.ones(24)))


def _drawer_cache():
    static_targets.set_endpoint_cache_for_tests({"base__drawer": {
        "ll": {"min": [0.0, 0.0, 0.0], "max": [0.4, 0.1, 0.1]},
        "ul": {"min": [0.0, -0.22, 0.0], "max": [0.4, -0.12, 0.1]},
    }})
    static_targets.set_joint_range_cache_for_tests({"base__drawer": {"ll": 0.0, "ul": 0.22}})


def test_fingertip_from_frames_extends_along_the_flange_axis():
    frames = np.zeros((9, 3))
    frames[7] = [0.0, 0.0, 0.5]
    frames[8] = [0.0, 0.0, 0.4]  # flange axis points down
    assert np.allclose(sm.fingertip_from_frames(frames), [0.0, 0.0, 0.4 - sm.GRIPPER_TIP_OFFSET])


def test_door_contact_offset_is_measured_from_successful_rollout_ends():
    _drawer_cache()
    base = np.asarray(sm.BASE_POSITION)
    # Rollout 0 succeeds and ends with the fingertip at world (0.2, -0.1, 0.05);
    # rollout 1 fails and must be ignored.
    tip_world = np.array([0.2, -0.1, 0.05])
    end = np.zeros((9, 3))
    end[8] = tip_world - base + [0, 0, sm.GRIPPER_TIP_OFFSET]
    end[7] = end[8] + [0, 0, 0.1]
    junk = np.full((9, 3), 9.0)
    junk[7, 2] = 10.0
    rollouts = {"frames": np.array([junk, end, junk]), "rollout_id": np.array([0, 0, 1]),
                "per_rollout_id": np.array([0, 1]), "per_rollout_reached": np.array([True, False])}

    offset, std, n = sm.door_contact_offset(rollouts, "base__drawer", q_end=0.11)

    assert n == 1
    assert np.allclose(offset, tip_world - np.array([0.0, -0.11, 0.0]))  # box_min at q=0.11
    assert np.allclose(std, 0.0)


def test_door_goals_use_the_measured_contact_point_when_given():
    _drawer_cache()
    offset = np.array([0.2, 0.05, 0.04])
    contact, completion = sm.task_goals(
        "open_drawer", ("move_door_rel", "base__drawer", 0.12), _scene_obs(), {"open_drawer": offset},
    )
    assert np.allclose(contact.box_min, contact.box_max)
    assert np.allclose(contact.box_min, _local(offset))
    assert np.allclose(completion.box_min, _local(np.array([0.0, -0.12, 0.0]) + offset))


def _arm_frames_along_x(n_steps, z=0.0):
    frames = np.zeros((n_steps, 9, 3))
    frames[:, :, 0] = np.arange(9) * 0.1
    frames[:, :, 2] = z
    return frames


def test_hit_rate_band_and_center_offset(monkeypatch):
    # Every candidate passes the start/target/planner checks, so only the
    # band and the offset decide.
    monkeypatch.setattr(sm, "arm_obstacle_clearance", lambda q, obstacles: 1.0)
    monkeypatch.setattr(sm, "check_feasibility",
                        lambda *a, **k: {"feasible": True, "attempts": [], "contact_config": [], "completion_config": []})
    frames = np.concatenate([_arm_frames_along_x(2, 0.0), _arm_frames_along_x(2, 1.0)])
    rollout_id = np.array([0, 0, 1, 1])
    on_path = np.array([[0.35, 0.0, 0.0]])  # touched by rollout 0 only -> hit rate 0.5
    goals = (sm.TaskGoal(wrist_turn=0.0), sm.TaskGoal(wrist_turn=0.0))

    obstacle, info = sm.sample_scene_obstacle(
        on_path, goals, 0.01, np.random.default_rng(0), np.random.default_rng(1),
        hit_rate_range=(0.0, 0.4), rollout_frames=frames, rollout_id=rollout_id, max_candidates=5,
    )
    assert obstacle is None and info["rejected"]["hit_rate"] == 5

    obstacle, _ = sm.sample_scene_obstacle(
        on_path, goals, 0.01, np.random.default_rng(0), np.random.default_rng(1),
        hit_rate_range=(0.4, 0.6), rollout_frames=frames, rollout_id=rollout_id,
    )
    assert np.allclose(obstacle.center, on_path[0])  # no offset by default

    obstacle, _ = sm.sample_scene_obstacle(
        on_path, goals, 0.01, np.random.default_rng(0), np.random.default_rng(1), center_offset_max=0.2,
    )
    moved = np.linalg.norm(obstacle.center - on_path[0])
    assert 0.0 < moved <= 0.2


def test_rollout_hit_rate_with_hand_sees_an_obstacle_only_the_fingers_reach():
    from shortstop.calvin.robot_geometry import panda_frames
    from shortstop.calvin.rollout_collection import tcp_from_frames
    q = np.array([0.0, 0.5, 0.0, -1.5, 0.0, 1.8, 0.7])
    frames = panda_frames(q)[None]
    tcp = tcp_from_frames(frames[0])
    direction = (tcp - frames[0][-1]) / np.linalg.norm(tcp - frames[0][-1])
    obstacle = Obstacle(center=tcp + 0.08 * direction, radius=0.05)  # past the fingertips
    rollout_id = np.array([0])
    assert sm.rollout_hit_rate(frames, rollout_id, obstacle) == 0.0
    assert sm.rollout_hit_rate(frames, rollout_id, obstacle, include_hand=True) == 1.0
