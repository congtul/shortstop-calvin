# CALVIN environment checks

Checks run on `mdt_env` (WSL2, calvin_env commit `797142c5`), 2026-10-04/05.

## scene_info fields (`scripts/verify_calvin_scene_info_fields.py`)

`env.get_info()["scene_info"]` has `buttons`, `doors`, `fixed_objects`,
`lights`, `movable_objects`, `switches`.

| Key | Entries | Fields used |
|---|---|---|
| `movable_objects` | `block_red`, `block_blue`, `block_pink` | `current_pos` (also `current_orn` xyzw, velocities, contacts) |
| `doors` | `base__slide`, `base__drawer` | `current_state` (joint value) |
| `switches` | `base__switch` | `joint_state`, `logical_state` |
| `buttons` | `base__button` | `joint_state`, `logical_state` |
| `lights` | `lightbulb`, `led` | `logical_state` only — read through the actuator (lightbulb ← switch, led ← button) |

Task config args (`task_fn.args`), e.g.:

```
open_drawer            ('base__drawer', 0.12)
move_slider_left       ('base__slide', 0.15)
push_blue_block_left   ('block_blue', -0.1, 0)
rotate_red_block_left  ('block_red', 60)          # degrees about z
lift_pink_block_slider ('block_pink', 0.03, 'table', 'plank_link')
turn_on_lightbulb      ('lightbulb', 0, 1)
```

## Determinism (`scripts/verify_calvin_env_determinism.py`)

The same chunk executed 240 times from identical snapshots gives arm frames
equal to within 1 µm (234/240 exactly 0; the rest are contact-solver noise
in `move_slider_left`). `env.step` is deterministic, so **w_bar = 0**.

This needed one fix: CALVIN's robot runs with `use_target_pose: true` and
accumulates every relative action into `robot.target_pos/target_orn`, which
its `serialize()` / `reset_from_storage()` don't save. A bare restore leaves
the IK target where a forked branch moved it (up to ~0.1 m drift), so
`experiment.snapshot_env` / `restore_env_snapshot` copy the target pose too
(regression test `test_snapshot_env_restores_in_place_mutated_robot_target_pose`).

## Action scale

CALVIN builds the env from the dataset's `.hydra/merged_config.yaml`, which
keeps `max_rel_pos = 0.02` and `magic_scaling_factor_pos = 1`. Ten random
actions (some |a| > 1; neither the policy nor CALVIN clips) moved
`robot.target_pos` by exactly `0.02 × Σa` (error 3e-9 m); the TCP itself
trailed the target by ~4 mm. `reach.CALVIN_ACTION_SCALE = 0.02`.

## Geometry

- `panda_frames` matches PyBullet's link frames to < 1e-5 m
  (`scripts/verify_robot_geometry_against_pybullet.py`).
- TCP = `gripper_tip_position` = CALVIN's `tcp_link`, equal to
  `robot_obs[0:3]` to < 1 µm; finger anchors checked by
  `scripts/verify_gripper_geometry_against_pybullet.py`.
- Mesh/URDF constants: `scripts/verify_robot_geometry_mesh_constants.py`.

## Static furniture (`scripts/precompute_static_target_sweeps.py`)

Joint ranges: `base__slide` 0–0.28, `base__drawer` 0–0.22, `base__switch`
0–0.088, `base__button` 0–0.02; `plank_link` fixed. All four joints are
prismatic, so the cached end boxes interpolate exactly.

## Model error (`scripts/calibrate_arm_model_error.py`, 2026-10-05)

One-step position error of the TCP-Jacobian prediction over all arm frames
and the TCP, 7036 real steps: median 2.4 mm, p99 11.8 mm →
`MODEL_ERROR = 0.01473`. TCP alone: median 0.85 mm, p99 8.9 mm (the former
flange-Jacobian model: 1.65 mm / 24.9 mm).
