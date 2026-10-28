import numpy as np

from shortstop.calvin.obstacle_viz import (
    _despawn_visual_obstacle, _spawn_visual_obstacle, _to_world_frame, save_sequence_video,
)
from shortstop.env import Obstacle


class _FakeBulletClient:
    """Records createVisualShape/createMultiBody/removeBody/stepSimulation
    calls instead of touching a real PyBullet simulation. This module's
    own unit tests only need to check WHAT gets spawned/despawned/replayed
    (lifecycle, and that obstacle.center/radius are used directly, no
    recomputation) -- there's no meaningful way to assert real rendered
    pixel content (shading/occlusion/perspective) without a real
    PyBullet renderer, which is an integration concern."""

    GEOM_SPHERE = "GEOM_SPHERE"

    def __init__(self):
        self.created_shapes = []
        self.created_bodies = []
        self.removed_body_ids = []
        self.step_count = 0
        self._next_shape_id = 100
        self._next_body_id = 200

    def createVisualShape(self, shapeType, radius=None, rgbaColor=None, physicsClientId=None):
        shape_id = self._next_shape_id
        self._next_shape_id += 1
        self.created_shapes.append({
            "shapeType": shapeType, "radius": radius, "rgbaColor": rgbaColor, "physicsClientId": physicsClientId,
        })
        return shape_id

    def createMultiBody(self, baseMass, baseCollisionShapeIndex, baseVisualShapeIndex, basePosition, physicsClientId=None):
        body_id = self._next_body_id
        self._next_body_id += 1
        self.created_bodies.append({
            "body_id": body_id, "baseMass": baseMass, "baseCollisionShapeIndex": baseCollisionShapeIndex,
            "baseVisualShapeIndex": baseVisualShapeIndex, "basePosition": basePosition,
            "physicsClientId": physicsClientId,
        })
        return body_id

    def removeBody(self, body_id, physicsClientId=None):
        self.removed_body_ids.append(body_id)

    def stepSimulation(self, physicsClientId=None):
        self.step_count += 1


class _FakeStorageTarget:
    """Stand-in for env.env.robot / env.env.scene -- only reset_from_storage
    is read by _replay_frame."""

    def __init__(self):
        self.reset_calls = []

    def reset_from_storage(self, data):
        self.reset_calls.append(data)


class _FakeCalvinEnv:
    """Stand-in for env.env -- only the attributes save_sequence_video /
    _spawn_visual_obstacle / _replay_frame actually read."""

    def __init__(self):
        self.p = _FakeBulletClient()
        self.cid = 0
        self.robot = _FakeStorageTarget()
        self.robot.base_position = (0.0, 0.0, 0.0)
        self.robot.base_orientation = (0.0, 0.0, 0.0, 1.0)
        self.scene = _FakeStorageTarget()
        self._render_count = 0

    def get_obs(self):
        self._render_count += 1
        frame = np.full((4, 4, 3), fill_value=min(self._render_count, 255), dtype=np.uint8)
        return {"rgb_obs": {"rgb_static": frame}}


class _FakeEnv:
    def __init__(self):
        self.env = _FakeCalvinEnv()


def _fake_state_snapshots(n):
    return [{"robot": {"step": i, "part": "robot"}, "scene": {"step": i, "part": "scene"}} for i in range(n)]


def test_to_world_frame_applies_translation_and_rotation():
    # identity orientation: pure translation.
    world = _to_world_frame([1.0, 2.0, 3.0], base_position=[-0.34, -0.46, 0.24], base_orientation=[0.0, 0.0, 0.0, 1.0])
    assert np.allclose(world, [0.66, 1.54, 3.24])

    # 90-degree yaw (matches basic_playtable.yaml's robot_base_orientation
    # convention): local +x should end up along world +y.
    quarter_turn_z = [0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]
    world = _to_world_frame([1.0, 0.0, 0.0], base_position=[0.0, 0.0, 0.0], base_orientation=quarter_turn_z)
    assert np.allclose(world, [0.0, 1.0, 0.0], atol=1e-9)


def test_spawn_visual_obstacle_uses_existing_center_and_radius_directly_no_collision():
    """The sphere must be positioned/sized from obstacle.center/radius
    directly -- no recomputation, no image-space projection -- and must
    be visual-only (baseCollisionShapeIndex=-1, per this task's own
    requirement 6: the existing capsule-chain clearance/shield logic
    stays the sole safety mechanism, this sphere never participates in
    physics)."""
    env = _FakeEnv()
    env.env.robot.base_position = [0.3, 0.15, 0.6]
    env.env.robot.base_orientation = [0.0, 0.0, 0.0, 1.0]
    obstacle = Obstacle(center=[1.0, 2.0, 3.0], radius=0.07)

    body_id = _spawn_visual_obstacle(env, obstacle)

    assert len(env.env.p.created_shapes) == 1
    assert env.env.p.created_shapes[0]["radius"] == 0.07
    assert len(env.env.p.created_bodies) == 1
    body = env.env.p.created_bodies[0]
    assert body["baseCollisionShapeIndex"] == -1
    assert body["baseMass"] == 0
    expected_world_center = _to_world_frame(obstacle.center, [0.3, 0.15, 0.6], [0.0, 0.0, 0.0, 1.0])
    assert np.allclose(body["basePosition"], expected_world_center)
    assert body_id == body["body_id"]  # _spawn_visual_obstacle returns createMultiBody's own return value


def test_despawn_removes_exactly_the_spawned_body():
    env = _FakeEnv()
    obstacle = Obstacle(center=[0.0, 0.0, 0.0], radius=0.05)

    body_id = _spawn_visual_obstacle(env, obstacle)
    _despawn_visual_obstacle(env, body_id)

    assert env.env.p.removed_body_ids == [body_id]


def test_save_sequence_video_writes_a_nonempty_file_with_and_without_obstacle(tmp_path):
    env = _FakeEnv()
    subtask_records = [
        {
            "subtask": "fake_subtask_a",
            "state_snapshots": _fake_state_snapshots(4),
            "obstacles": [Obstacle(center=[0.0, 0.0, 0.0], radius=0.05)],
            "outcome": "violated",
        },
        {
            "subtask": "fake_subtask_b",
            "state_snapshots": _fake_state_snapshots(3),
            "obstacles": None,
            "outcome": "reached",
        },
    ]
    out_path = tmp_path / "sequence.mp4"

    save_sequence_video(env, subtask_records, str(out_path), fps=5, freeze_frames=2)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_save_sequence_video_duration_includes_freeze_frames(tmp_path):
    """Checks the written MP4's duration matches the total number of
    logical frames requested -- (2 + 2 freeze) + (3 + 2 freeze) = 9
    frames at fps=5 -> 1.8s -- not just that some file got written."""
    try:
        from moviepy import VideoFileClip
    except ImportError:
        from moviepy.editor import VideoFileClip

    env = _FakeEnv()
    subtask_records = [
        {"subtask": "a", "state_snapshots": _fake_state_snapshots(2), "obstacles": None, "outcome": "reached"},
        {"subtask": "b", "state_snapshots": _fake_state_snapshots(3), "obstacles": None, "outcome": "failed"},
    ]
    out_path = tmp_path / "sequence.mp4"
    save_sequence_video(env, subtask_records, str(out_path), fps=5, freeze_frames=2)

    with VideoFileClip(str(out_path)) as clip:
        duration = clip.duration
    assert np.isclose(duration, 9 / 5, atol=0.15)


def test_save_sequence_video_replays_every_snapshot_via_reset_from_storage(tmp_path):
    """Each subtask's own state_snapshots must be replayed in order via
    robot.reset_from_storage/scene.reset_from_storage (see _replay_frame's
    own docstring) -- not skipped, not reordered, not substituted with a
    single snapshot repeated."""
    env = _FakeEnv()
    subtask_records = [
        {"subtask": "a", "state_snapshots": _fake_state_snapshots(3), "obstacles": None, "outcome": "reached"},
    ]
    out_path = tmp_path / "sequence.mp4"

    save_sequence_video(env, subtask_records, str(out_path), fps=5, freeze_frames=0)

    assert [c["step"] for c in env.env.robot.reset_calls] == [0, 1, 2]
    assert [c["step"] for c in env.env.scene.reset_calls] == [0, 1, 2]


def test_save_sequence_video_spawns_and_despawns_exactly_once_per_subtask_with_an_obstacle():
    """Lifecycle: a sphere is spawned once per subtask that HAS an
    obstacle and despawned before moving to the next subtask -- never left
    to accumulate across subtasks/sequences (this task's own requirement
    7)."""
    env = _FakeEnv()
    subtask_records = [
        {
            "subtask": "a", "state_snapshots": _fake_state_snapshots(2),
            "obstacles": [Obstacle(center=[0.0, 0.0, 0.0], radius=0.05)], "outcome": "violated",
        },
        {"subtask": "b", "state_snapshots": _fake_state_snapshots(2), "obstacles": None, "outcome": "reached"},
        {
            "subtask": "c", "state_snapshots": _fake_state_snapshots(2),
            "obstacles": [Obstacle(center=[1.0, 0.0, 0.0], radius=0.05)], "outcome": "reached",
        },
    ]
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir) / "sequence.mp4"
        save_sequence_video(env, subtask_records, str(out_path), fps=5, freeze_frames=0)

    assert len(env.env.p.created_bodies) == 2  # subtasks a and c only, not b
    created_ids = [body["body_id"] for body in env.env.p.created_bodies]
    # every created body was later removed, none leaked, none duplicated
    assert sorted(env.env.p.removed_body_ids) == sorted(created_ids)
