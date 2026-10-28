"""Offline pieces of scripts/collect_calvin_unshielded_rollouts.py: turn
recorded unshielded rollouts of one task into
- a workspace point cloud (per step, the 9 arm frames plus points inside
  each link, tagged with their radius) -- where scene obstacles are drawn;
- the raw per-step frames (e.g. for the measured door contact points).
"""
import json
import os

import numpy as np

from .robot_geometry import FLANGE_FRAME_INDEX, FRAME_RADIUS, GRIPPER_TIP_OFFSET, GRIPPER_TIP_RADIUS, LINK_RADIUS

# Dropped from the task set (no positional goal).
EXCLUDED_TASKS = ("stack_block", "unstack_block")


def eligible_task_names(tasks, initial_states, valid_task_fn):
    """(eligible, chain_only): task names, sorted, that can / cannot be the
    FIRST task from at least one enumerated initial state, excluding
    EXCLUDED_TASKS. A single-task trial can only start a task from a fresh
    initial state, so chain_only tasks (e.g. lift_*_block_drawer,
    place_in_* -- they need a block already grasped/placed by an earlier
    subtask) can't be collected this way."""
    eligible, chain_only = [], []
    for name in sorted(tasks):
        if name in EXCLUDED_TASKS:
            continue
        if any(valid_task_fn(state, tasks[name]) for state in initial_states):
            eligible.append(name)
        else:
            chain_only.append(name)
    return eligible, chain_only


def trials_for_task(task_name, tasks, initial_states, valid_task_fn, num_trials, rng_seed):
    """`num_trials` (initial_state, (task_name,)) pairs drawn with
    replacement from the initial states where `task_name` is valid, with
    its own RandomState.

    Prefix-stable: drawn one index at a time, so trial i is the same for
    any num_trials > i. Checkpoint resume relies on this -- raising
    N_ROLLOUTS later reuses rollouts 0..old_N-1 unchanged and only adds new
    ones."""
    valid_states = [state for state in initial_states if valid_task_fn(state, tasks[task_name])]
    if not valid_states:
        raise ValueError(f"task_name={task_name!r} is never valid for any enumerated initial_state")
    rng = np.random.RandomState(rng_seed)
    chosen = [rng.randint(len(valid_states)) for _ in range(num_trials)]
    return [(valid_states[i], (task_name,)) for i in chosen]


def link_interior_samples(frame_a, frame_b, n_interior):
    """n_interior evenly-spaced points strictly between frame_a and frame_b
    (both endpoints excluded -- they're already their own frame points)."""
    if n_interior <= 0:
        return np.empty((0, 3))
    ts = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
    return frame_a[None, :] + ts[:, None] * (frame_b - frame_a)[None, :]


def raw_rollout_arrays(records):
    """Flattens per-rollout records ({"rollout_id", "reached",
    "steps_taken", "trajectory": [(9, 3) per step, incl. the pre-action
    start]}) into arrays for the raw .npz -- everything both derived
    datasets need, so either can be rebuilt later without re-simulating."""
    frames, rollout_id, step_index = [], [], []
    for record in records:
        for i, step_frames in enumerate(record["trajectory"]):
            frames.append(np.asarray(step_frames, dtype=float))
            rollout_id.append(record["rollout_id"])
            step_index.append(i)
    return {
        "frames": np.asarray(frames, dtype=float).reshape(-1, FLANGE_FRAME_INDEX + 1, 3),
        "rollout_id": np.asarray(rollout_id, dtype=int),
        "step_index": np.asarray(step_index, dtype=int),
        "per_rollout_id": np.asarray([r["rollout_id"] for r in records], dtype=int),
        "per_rollout_reached": np.asarray([bool(r["reached"]) for r in records], dtype=bool),
        "per_rollout_steps_taken": np.asarray([r["steps_taken"] for r in records], dtype=int),
    }


def tcp_from_frames(frames):
    """TCP from panda_frames() rows (..., 9, 3): GRIPPER_TIP_OFFSET past the
    flange along the frame 7 -> 8 direction (the flange z axis)."""
    frames = np.asarray(frames, dtype=float)
    z = frames[..., 8, :] - frames[..., 7, :]
    return frames[..., 8, :] + GRIPPER_TIP_OFFSET * z / np.linalg.norm(z, axis=-1, keepdims=True)


def records_from_raw(raw):
    """Per-rollout records ({rollout_id, reached, trajectory}) rebuilt from a
    rollouts_<task>.npz (raw_rollout_arrays' output)."""
    reached = dict(zip(raw["per_rollout_id"].tolist(), raw["per_rollout_reached"].tolist()))
    return [{"rollout_id": int(k), "reached": bool(reached[k]),
             "trajectory": raw["frames"][raw["rollout_id"] == k]} for k in raw["per_rollout_id"]]


def workspace_arrays(records, link_samples_per_segment, only_successful, include_hand=False):
    """The workspace point cloud: per step, the 9 frame points (tagged
    frame_index 0..8, FRAME_RADIUS) plus `link_samples_per_segment`
    interior points on each of the 8 links (tagged link_index 0..7,
    LINK_RADIUS). `include_hand` adds the TCP and the same number of interior
    points on the flange -> TCP segment (link_index 8, GRIPPER_TIP_RADIUS).
    `None` if no rollout qualifies."""
    points, frame_index, link_index, radius, rollout_id = [], [], [], [], []
    for record in records:
        if only_successful and not record["reached"]:
            continue
        for frames in record["trajectory"]:
            frames = np.asarray(frames, dtype=float)
            for i in range(FLANGE_FRAME_INDEX + 1):
                points.append(frames[i])
                frame_index.append(i)
                link_index.append(-1)
                radius.append(FRAME_RADIUS[i])
                rollout_id.append(record["rollout_id"])
            for i in range(FLANGE_FRAME_INDEX):
                for point in link_interior_samples(frames[i], frames[i + 1], link_samples_per_segment):
                    points.append(point)
                    frame_index.append(-1)
                    link_index.append(i)
                    radius.append(LINK_RADIUS[i])
                    rollout_id.append(record["rollout_id"])
            if include_hand:
                tcp = tcp_from_frames(frames)
                hand = [tcp, *link_interior_samples(frames[FLANGE_FRAME_INDEX], tcp, link_samples_per_segment)]
                for point in hand:
                    points.append(point)
                    frame_index.append(-1)
                    link_index.append(FLANGE_FRAME_INDEX)
                    radius.append(GRIPPER_TIP_RADIUS)
                    rollout_id.append(record["rollout_id"])
    if not points:
        return None
    return {
        "points": np.asarray(points, dtype=float),
        "frame_index": np.asarray(frame_index, dtype=int),
        "link_index": np.asarray(link_index, dtype=int),
        "radius": np.asarray(radius, dtype=float),
        "rollout_id": np.asarray(rollout_id, dtype=int),
    }


# --- Checkpointing (one file per finished rollout) ---------------------------
#
# A full collection (27 tasks x 50 rollouts) takes hours, so every finished
# rollout is saved on its own and a re-run skips whatever is already on disk.
# Results match an uninterrupted run because a rollout depends only on its
# rollout_id (prefix-stable trial draw + per-id seed). A manifest pins the
# settings that change what a rollout IS; resuming under different ones
# raises instead of silently mixing incompatible rollouts.

MANIFEST_NAME = "manifest.json"


def rollout_checkpoint_path(checkpoint_dir, task_name, rollout_id):
    return os.path.join(checkpoint_dir, task_name, f"rollout_{rollout_id:04d}.npz")


def save_rollout_checkpoint(path, record):
    """Writes one rollout record atomically (temp file + rename), so an
    interrupted write never leaves a half-file that later loads as data."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp.npz"
    np.savez(
        tmp_path,
        rollout_id=int(record["rollout_id"]),
        reached=bool(record["reached"]),
        steps_taken=int(record["steps_taken"]),
        trajectory=np.asarray(record["trajectory"], dtype=float).reshape(-1, FLANGE_FRAME_INDEX + 1, 3),
    )
    os.replace(tmp_path, path)


def load_rollout_checkpoint(path):
    with np.load(path) as data:
        return {
            "rollout_id": int(data["rollout_id"]),
            "reached": bool(data["reached"]),
            "steps_taken": int(data["steps_taken"]),
            "trajectory": list(np.asarray(data["trajectory"], dtype=float)),
        }


def check_or_write_manifest(checkpoint_dir, manifest):
    """First run: writes `manifest` (a JSON-able dict). Later runs: raises
    ValueError if any key differs from what's stored, naming the keys."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, MANIFEST_NAME)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        return
    with open(path, encoding="utf-8") as f:
        stored = json.load(f)
    diff = sorted(k for k in set(stored) | set(manifest) if stored.get(k) != manifest.get(k))
    if diff:
        details = ", ".join(f"{k}: stored={stored.get(k)!r} now={manifest.get(k)!r}" for k in diff)
        raise ValueError(
            f"checkpoint dir {checkpoint_dir} was collected with different settings ({details}) -- "
            f"use a new CHECKPOINT_DIR or delete the old one"
        )
