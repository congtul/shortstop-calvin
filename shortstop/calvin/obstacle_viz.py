"""Debug videos of finished rollouts with the obstacle drawn in.

The obstacle is invisible to the policy. After a rollout has finished, its
recorded `state_snapshots` are replayed into the same env one at a time and
re-rendered from the static camera, with the obstacle spawned as a
visual-only PyBullet sphere (real perspective and occlusion, no collision).
Replay only resets body states; it never calls the policy or env.step().
"""
from pathlib import Path

import cv2

from .experiment import _clearance_breakdown, restore_env_snapshot
from .robot_geometry import to_world_frame as _to_world_frame


def _import_image_sequence_clip():
    """moviepy < 2.0 has it under moviepy.editor, >= 2.0 at the top level."""
    try:
        from moviepy.editor import ImageSequenceClip
    except ImportError:
        from moviepy import ImageSequenceClip
    return ImageSequenceClip


_OBSTACLE_RGBA = (1.0, 0.0, 0.0, 1.0)


def _spawn_visual_obstacle(env, obstacle):
    """Visual-only sphere at the obstacle (local frame -> world); returns its id."""
    p, cid = env.env.p, env.env.cid
    world_center = _to_world_frame(obstacle.center, env.env.robot.base_position, env.env.robot.base_orientation)
    visual_shape = p.createVisualShape(
        p.GEOM_SPHERE, radius=obstacle.radius, rgbaColor=_OBSTACLE_RGBA, physicsClientId=cid,
    )
    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=visual_shape,
        basePosition=world_center,
        physicsClientId=cid,
    )


def _despawn_visual_obstacle(env, body_id):
    """Remove a sphere created by `_spawn_visual_obstacle`."""
    env.env.p.removeBody(body_id, physicsClientId=env.env.cid)


def _replay_frame(env, state_snapshot):
    """Restore one recorded snapshot and render rgb_static."""
    restore_env_snapshot(env, state_snapshot)
    return env.env.get_obs()["rgb_obs"]["rgb_static"]


def save_sequence_video(env, subtask_records, out_path, fps=10, freeze_frames=5, save_violation_frame_dir=None):
    """Write one MP4 for a rollout's subtask attempts, holding each attempt's
    last frame `freeze_frames` times.

    `subtask_records`: [{subtask, state_snapshots, obstacles, outcome}] from
    a harness run with record_trajectory=True, replayed into the same `env`.
    `save_violation_frame_dir`: also save the last frame of each violated
    attempt as PNG and print the per-primitive clearances."""
    ImageSequenceClip = _import_image_sequence_clip()

    frames = []
    for i, record in enumerate(subtask_records):
        obstacles = record["obstacles"] or []
        body_ids = [_spawn_visual_obstacle(env, o) for o in obstacles]
        try:
            for state_snapshot in record["state_snapshots"]:
                frames.append(_replay_frame(env, state_snapshot))
        finally:
            for body_id in body_ids:
                _despawn_visual_obstacle(env, body_id)
        if save_violation_frame_dir is not None and record["outcome"] == "violated":
            save_violation_frame_dir.mkdir(parents=True, exist_ok=True)
            image_path = (
                save_violation_frame_dir / f"{Path(out_path).stem}_subtask{i}_{record['subtask']}_violated.png"
            )
            cv2.imwrite(str(image_path), cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR))

            breakdown = _clearance_breakdown(env.get_obs(), obstacles)
            min_label, min_value = min(breakdown, key=lambda item: item[1])
            print(f"[violation] {image_path.name}: min_clearance={min_value:.4f} at {min_label}")
            for label, value in breakdown:
                print(f"    {label}: {value:.4f}")
        frames.extend([frames[-1]] * freeze_frames)

    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(str(out_path), codec="libx264", audio=False, logger=None)
