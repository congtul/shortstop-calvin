"""Precomputed geometry of task_D_D's static furniture (slider, drawer,
switch, button), read by task_goals.

- CACHE_PATH (scripts/precompute_static_target_sweeps.py): for each joint,
  the world-frame AABB of its moving link at the two ends of its travel and
  the joint range (ll, ul). Every joint is prismatic, so the box at any q is
  an exact linear interpolation (`static_target_box_at`).
- DOOR_CONTACT_PATH (scripts/build_door_contact_offsets.py): per
  move_door_rel task, where the fingertip works the door -- an offset from
  the link box's min corner, measured on successful rollouts (the link is
  the whole panel; the hand only uses the handle or the front face).

The furniture never moves between episodes (single scene D), so both are
built once offline; this module only loads them (json/numpy only).
"""
import json
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"
CACHE_PATH = DATA_DIR / "calvin_task_D_D_target_sweeps.json"
DOOR_CONTACT_PATH = DATA_DIR / "door_contact_offsets.json"

# Lazily loaded once per process.
_endpoints_cache = None
_joint_ranges_cache = None
_door_contact_cache = None


def _read_json(path):
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _endpoints():
    global _endpoints_cache
    if _endpoints_cache is None:
        _endpoints_cache = _read_json(CACHE_PATH).get("targets_world_frame_endpoints", {})
    return _endpoints_cache


def _joint_ranges():
    global _joint_ranges_cache
    if _joint_ranges_cache is None:
        generated_from = _read_json(CACHE_PATH).get("generated_from", {})
        _joint_ranges_cache = {name: e for name, e in generated_from.items() if "ll" in e and "ul" in e}
    return _joint_ranges_cache


def static_target_endpoint_boxes(name):
    """((min, max) at ll, (min, max) at ul) for cache key `name`, or None."""
    entry = _endpoints().get(name)
    if entry is None:
        return None
    return tuple((np.asarray(entry[end]["min"], dtype=float), np.asarray(entry[end]["max"], dtype=float))
                 for end in ("ll", "ul"))


def static_target_joint_range(name):
    """(ll, ul) of `name`'s joint, or None."""
    entry = _joint_ranges().get(name)
    if entry is None:
        return None
    return float(entry["ll"]), float(entry["ul"])


def static_target_box_at(name, q):
    """World-frame (box_min, box_max) of `name` at joint value `q` (clipped
    to the travel), or None if not cached."""
    endpoints = static_target_endpoint_boxes(name)
    joint_range = static_target_joint_range(name)
    if endpoints is None or joint_range is None or joint_range[0] == joint_range[1]:
        return None
    ll, ul = joint_range
    frac = float(np.clip((q - ll) / (ul - ll), 0.0, 1.0))
    (ll_min, ll_max), (ul_min, ul_max) = endpoints
    return ll_min + frac * (ul_min - ll_min), ll_max + frac * (ul_max - ll_max)


def door_contact_offset(task_name):
    """(3,) fingertip offset from the door link's box_min, or None."""
    global _door_contact_cache
    if _door_contact_cache is None:
        _door_contact_cache = {k: v["offset_from_box_min"] for k, v in _read_json(DOOR_CONTACT_PATH).items()}
    offset = _door_contact_cache.get(task_name)
    return None if offset is None else np.asarray(offset, dtype=float)


def reset_cache_for_tests():
    global _endpoints_cache, _joint_ranges_cache, _door_contact_cache
    _endpoints_cache = _joint_ranges_cache = _door_contact_cache = None


def set_endpoint_cache_for_tests(endpoints):
    """{name: {"ll": {"min", "max"}, "ul": {"min", "max"}}}"""
    global _endpoints_cache
    _endpoints_cache = endpoints


def set_joint_range_cache_for_tests(joint_ranges):
    """{name: {"ll": float, "ul": float}}"""
    global _joint_ranges_cache
    _joint_ranges_cache = joint_ranges


def set_door_contact_cache_for_tests(offsets):
    """{task_name: (3,)}"""
    global _door_contact_cache
    _door_contact_cache = offsets
