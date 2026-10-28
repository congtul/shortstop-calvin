import numpy as np
import pytest

from shortstop.calvin.robot_geometry import FLANGE_FRAME_INDEX, FRAME_RADIUS, LINK_RADIUS
from shortstop.calvin.rollout_collection import (
    check_or_write_manifest, eligible_task_names, link_interior_samples, load_rollout_checkpoint,
    raw_rollout_arrays, rollout_checkpoint_path, save_rollout_checkpoint,
    trials_for_task, workspace_arrays,
)


def _frames(offset):
    """(9, 3) frames, frame i at (i + offset, 0, 0) -- easy to tell apart."""
    frames = np.zeros((FLANGE_FRAME_INDEX + 1, 3))
    frames[:, 0] = np.arange(FLANGE_FRAME_INDEX + 1) + offset
    return frames


def _record(rollout_id, reached, n_steps):
    # trajectory includes the pre-action start, so n_steps + 1 entries
    return {
        "rollout_id": rollout_id, "reached": reached, "steps_taken": n_steps,
        "trajectory": [_frames(100 * rollout_id + step) for step in range(n_steps + 1)],
    }


def _valid(state, task):
    return task in state["allows"]


def test_eligible_task_names_drops_excluded_and_splits_chain_only():
    tasks = {"open_drawer": "a", "stack_block": "b", "unstack_block": "c", "place_in_drawer": "d"}
    states = [{"allows": {"a", "b"}}, {"allows": {"a"}}]

    eligible, chain_only = eligible_task_names(tasks, states, _valid)

    assert eligible == ["open_drawer"]
    assert chain_only == ["place_in_drawer"]  # stack/unstack never listed, even as chain-only


def test_trials_for_task_only_draws_valid_states_and_is_seeded():
    tasks = {"open_drawer": "a"}
    states = [{"allows": {"a"}, "id": 0}, {"allows": set(), "id": 1}, {"allows": {"a"}, "id": 2}]

    trials = trials_for_task("open_drawer", tasks, states, _valid, num_trials=20, rng_seed=0)

    assert len(trials) == 20
    assert {state["id"] for state, _ in trials} <= {0, 2}
    assert all(seq == ("open_drawer",) for _, seq in trials)
    again = trials_for_task("open_drawer", tasks, states, _valid, num_trials=20, rng_seed=0)
    assert [s["id"] for s, _ in trials] == [s["id"] for s, _ in again]

    with pytest.raises(ValueError):
        trials_for_task("open_drawer", tasks, [{"allows": set()}], _valid, num_trials=1, rng_seed=0)


def test_link_interior_samples_excludes_endpoints():
    samples = link_interior_samples(np.zeros(3), np.array([4.0, 0.0, 0.0]), 3)
    assert np.allclose(samples[:, 0], [1.0, 2.0, 3.0])
    assert link_interior_samples(np.zeros(3), np.ones(3), 0).shape == (0, 3)


def test_raw_rollout_arrays_keeps_every_rollout_and_step():
    records = [_record(0, True, 2), _record(1, False, 1)]

    raw = raw_rollout_arrays(records)

    assert raw["frames"].shape == (5, FLANGE_FRAME_INDEX + 1, 3)
    assert raw["rollout_id"].tolist() == [0, 0, 0, 1, 1]
    assert raw["step_index"].tolist() == [0, 1, 2, 0, 1]
    assert raw["per_rollout_reached"].tolist() == [True, False]
    assert raw["per_rollout_steps_taken"].tolist() == [2, 1]


def test_workspace_arrays_tags_frames_and_links_and_filters_failures():
    records = [_record(0, True, 1), _record(1, False, 1)]

    ws = workspace_arrays(records, link_samples_per_segment=2, only_successful=True)

    per_step = (FLANGE_FRAME_INDEX + 1) + FLANGE_FRAME_INDEX * 2
    assert len(ws["points"]) == 2 * per_step  # only rollout 0, 2 trajectory entries
    assert set(ws["rollout_id"].tolist()) == {0}
    frame_rows = ws["frame_index"] >= 0
    assert np.allclose(ws["radius"][frame_rows], np.asarray(FRAME_RADIUS)[ws["frame_index"][frame_rows]])
    assert np.allclose(ws["radius"][~frame_rows], np.asarray(LINK_RADIUS)[ws["link_index"][~frame_rows]])

    with_failures = workspace_arrays(records, link_samples_per_segment=2, only_successful=False)
    assert len(with_failures["points"]) == 4 * per_step
    assert workspace_arrays([_record(0, False, 1)], 2, only_successful=True) is None


def test_trials_for_task_is_prefix_stable_so_raising_n_reuses_old_rollouts():
    tasks = {"open_drawer": "a"}
    states = [{"allows": {"a"}, "id": i} for i in range(7)]

    short = trials_for_task("open_drawer", tasks, states, _valid, num_trials=10, rng_seed=3)
    long = trials_for_task("open_drawer", tasks, states, _valid, num_trials=30, rng_seed=3)

    assert [s["id"] for s, _ in long[:10]] == [s["id"] for s, _ in short]


def test_rollout_checkpoint_round_trips(tmp_path):
    record = _record(7, True, 3)
    path = rollout_checkpoint_path(str(tmp_path), "open_drawer", 7)

    save_rollout_checkpoint(path, record)
    loaded = load_rollout_checkpoint(path)

    assert path.endswith("open_drawer/rollout_0007.npz")
    assert not list(tmp_path.rglob("*.tmp.npz"))
    assert (loaded["rollout_id"], loaded["reached"], loaded["steps_taken"]) == (7, True, 3)
    assert np.allclose(np.asarray(loaded["trajectory"]), np.asarray(record["trajectory"]))
    # Feeds the derived builders exactly like a fresh record.
    assert np.allclose(raw_rollout_arrays([loaded])["frames"], raw_rollout_arrays([record])["frames"])


def test_manifest_is_written_once_and_rejects_different_settings(tmp_path):
    check_or_write_manifest(str(tmp_path), {"ep_len": 360, "rollout_seed_base": 2_000_000})
    check_or_write_manifest(str(tmp_path), {"ep_len": 360, "rollout_seed_base": 2_000_000})  # same: ok

    with pytest.raises(ValueError, match="ep_len"):
        check_or_write_manifest(str(tmp_path), {"ep_len": 80, "rollout_seed_base": 2_000_000})


def test_tcp_from_frames_matches_forward_kinematics():
    from shortstop.calvin.robot_geometry import gripper_tip_position, panda_frames
    from shortstop.calvin.rollout_collection import tcp_from_frames
    q = np.array([0.0, 0.5, 0.0, -1.5, 0.0, 1.8, 0.7])
    assert np.allclose(tcp_from_frames(panda_frames(q)), gripper_tip_position(q))


def test_workspace_hand_points_lie_on_the_flange_to_tcp_segment():
    from shortstop.calvin.robot_geometry import GRIPPER_TIP_RADIUS, panda_frames
    from shortstop.calvin.rollout_collection import tcp_from_frames, workspace_arrays
    q = np.array([0.0, 0.5, 0.0, -1.5, 0.0, 1.8, 0.7])
    frames = panda_frames(q)
    record = {"rollout_id": 0, "reached": True, "trajectory": [frames]}
    without = workspace_arrays([record], 2, only_successful=True)
    with_hand = workspace_arrays([record], 2, only_successful=True, include_hand=True)
    hand = with_hand["link_index"] == FLANGE_FRAME_INDEX
    assert len(with_hand["points"]) == len(without["points"]) + 3  # TCP + 2 interior samples
    assert np.allclose(with_hand["radius"][hand], GRIPPER_TIP_RADIUS)
    tcp, flange = tcp_from_frames(frames), frames[FLANGE_FRAME_INDEX]
    for point in with_hand["points"][hand]:
        assert np.isclose(np.linalg.norm(point - flange) + np.linalg.norm(tcp - point), np.linalg.norm(tcp - flange))
