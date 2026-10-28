import json

import numpy as np
import pytest

from shortstop.calvin import static_targets


@pytest.fixture(autouse=True)
def _reset_cache():
    static_targets.reset_cache_for_tests()
    yield
    static_targets.reset_cache_for_tests()


DRAWER_CACHE = {
    "generated_from": {"base__drawer": {"ll": 0.0, "ul": 0.12}, "plank_link": {"fixed": True}},
    "targets_world_frame_endpoints": {
        "base__drawer": {
            "ll": {"min": [0.0, 0.1, 0.2], "max": [0.55, 0.225, 0.3]},
            "ul": {"min": [0.12, 0.1, 0.2], "max": [0.67, 0.225, 0.6]},
        },
    },
}


@pytest.fixture
def drawer_cache(tmp_path, monkeypatch):
    path = tmp_path / "sweeps.json"
    path.write_text(json.dumps(DRAWER_CACHE), encoding="utf-8")
    monkeypatch.setattr(static_targets, "CACHE_PATH", path)
    return path


def test_everything_is_none_without_a_cache_file(tmp_path, monkeypatch):
    monkeypatch.setattr(static_targets, "CACHE_PATH", tmp_path / "missing.json")
    monkeypatch.setattr(static_targets, "DOOR_CONTACT_PATH", tmp_path / "missing2.json")
    assert static_targets.static_target_endpoint_boxes("base__drawer") is None
    assert static_targets.static_target_joint_range("base__drawer") is None
    assert static_targets.static_target_box_at("base__drawer", 0.05) is None
    assert static_targets.door_contact_offset("open_drawer") is None


def test_loads_endpoint_boxes_and_joint_range(drawer_cache):
    (ll_min, ll_max), (ul_min, ul_max) = static_targets.static_target_endpoint_boxes("base__drawer")
    assert np.allclose(ll_min, [0.0, 0.1, 0.2]) and np.allclose(ll_max, [0.55, 0.225, 0.3])
    assert np.allclose(ul_min, [0.12, 0.1, 0.2]) and np.allclose(ul_max, [0.67, 0.225, 0.6])
    assert static_targets.static_target_joint_range("base__drawer") == (0.0, 0.12)
    # Fixed links have no travel.
    assert static_targets.static_target_joint_range("plank_link") is None
    assert static_targets.static_target_endpoint_boxes("plank_link") is None


def test_box_at_interpolates_and_clips(drawer_cache):
    box_min, box_max = static_targets.static_target_box_at("base__drawer", 0.06)
    assert np.allclose(box_min, [0.06, 0.1, 0.2])
    assert np.allclose(box_max, [0.61, 0.225, 0.45])
    beyond_min, _ = static_targets.static_target_box_at("base__drawer", 1.0)
    assert np.allclose(beyond_min, [0.12, 0.1, 0.2])


def test_cache_is_read_once_per_process(drawer_cache):
    assert static_targets.static_target_joint_range("base__drawer") == (0.0, 0.12)
    drawer_cache.write_text(json.dumps({"generated_from": {"base__drawer": {"ll": 0.0, "ul": 9.0}}}), encoding="utf-8")
    assert static_targets.static_target_joint_range("base__drawer") == (0.0, 0.12)


def test_door_contact_offset_reads_the_file(tmp_path, monkeypatch):
    path = tmp_path / "doors.json"
    path.write_text(json.dumps({"open_drawer": {"offset_from_box_min": [0.1, 0.2, 0.3]}}), encoding="utf-8")
    monkeypatch.setattr(static_targets, "DOOR_CONTACT_PATH", path)
    assert np.allclose(static_targets.door_contact_offset("open_drawer"), [0.1, 0.2, 0.3])
    assert static_targets.door_contact_offset("close_drawer") is None


def test_setters_bypass_the_files(tmp_path, monkeypatch):
    monkeypatch.setattr(static_targets, "CACHE_PATH", tmp_path / "missing.json")
    static_targets.set_endpoint_cache_for_tests({
        "led": {"ll": {"min": [0, 0, 0], "max": [0, 0, 0]}, "ul": {"min": [1, 1, 1], "max": [1, 1, 1]}},
    })
    static_targets.set_joint_range_cache_for_tests({"led": {"ll": 0.0, "ul": 1.0}})
    box_min, _ = static_targets.static_target_box_at("led", 0.5)
    assert np.allclose(box_min, [0.5, 0.5, 0.5])
