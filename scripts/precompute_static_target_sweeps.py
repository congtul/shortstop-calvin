"""Precompute shortstop/calvin/data/calvin_task_D_D_target_sweeps.json: the
world-frame AABB of each static furniture part at both ends of its travel,
read from CALVIN's PyBullet scene.

- base__slide, base__drawer: the moving link of each door;
- lightbulb / led: their actuator (base__switch / base__button), since the
  hand works the actuator, not the light;
- plank_link: fixed, a single box.

All four joints are prismatic (checked at sweep time by `_assert_prismatic`),
so the box at any joint value is an exact interpolation of the two end boxes
(static_targets.static_target_box_at), and the swept union is their hull.
Run once; only needed again if the scene assets change.

Run from WSL2, inside `mdt_env`:

    cd SHORTSTOP
    python scripts/precompute_static_target_sweeps.py
"""
import itertools
import json
import sys
from pathlib import Path

import calvin_env  # noqa: F401 -- see the sys.path comment below
import hydra
import numpy as np
from pytorch_lightning import seed_everything

REPO_ROOT = Path(__file__).resolve().parents[1]
MDT_POLICY_ROOT = REPO_ROOT / "mdt_policy"
sys.path.insert(0, str(REPO_ROOT))
# calvin_env must be imported before mdt_policy/ goes on sys.path (it contains a calvin_env/ dir).
sys.path.insert(0, str(MDT_POLICY_ROOT))

from mdt.evaluation.multistep_sequences import get_sequences  # noqa: E402
from mdt.evaluation.utils import get_env_state_for_initial_condition  # noqa: E402

from shortstop.calvin.baseline_runner import setup_env_and_policy  # noqa: E402
from shortstop.calvin.config import N_CANDIDATES  # noqa: E402
from shortstop.calvin.experiment import restore_env_snapshot, snapshot_env  # noqa: E402
from shortstop.calvin.static_targets import CACHE_PATH  # noqa: E402

DOOR_JOINT_NAMES = ("base__slide", "base__drawer")
LIGHT_NAMES = ("lightbulb", "led")


def _assert_prismatic(p, cid, uid, joint_index, cache_key):
    joint_type = p.getJointInfo(uid, joint_index, physicsClientId=cid)[2]
    if joint_type != p.JOINT_PRISMATIC:
        raise RuntimeError(
            f"{cache_key!r} (uid={uid}, joint_index={joint_index}) is joint type {joint_type}, "
            "not Prismatic -- the exact 2-endpoint AABB-union shortcut this script uses is only "
            "valid for a pure straight-line translation (see this script's own module docstring). "
            "A genuinely Revolute/Spherical joint here would need real curvature-aware sampling "
            "instead -- do not silently reuse this function for it."
        )


def _real_aabb(p, cid, uid, link_index):
    box_min, box_max = p.getAABB(uid, link_index, physicsClientId=cid)
    return np.asarray(box_min, dtype=float), np.asarray(box_max, dtype=float)


def _swept_box(p, cid, uid, joint_index, cache_key):
    """(swept union box, (box at ll, box at ul)) of a prismatic joint's moving link."""
    _assert_prismatic(p, cid, uid, joint_index, cache_key)
    ll, ul = p.getJointInfo(uid, joint_index, physicsClientId=cid)[8:10]

    p.resetJointState(uid, joint_index, float(ll), physicsClientId=cid)
    min_at_ll, max_at_ll = _real_aabb(p, cid, uid, joint_index)
    p.resetJointState(uid, joint_index, float(ul), physicsClientId=cid)
    min_at_ul, max_at_ul = _real_aabb(p, cid, uid, joint_index)

    box_min = np.minimum(min_at_ll, min_at_ul)
    box_max = np.maximum(max_at_ll, max_at_ul)
    return box_min, box_max, float(ll), float(ul), (min_at_ll, max_at_ll), (min_at_ul, max_at_ul)


def compute_static_target_sweeps(env):
    """Runs the full precomputation against a REAL, already-reset CALVIN
    env -- snapshot/restore (same idiom shortstop.calvin.experiment.
    _privileged_rollout_violates already uses) so every resetJointState
    call this makes along the way leaves the env's real state exactly as
    it found it. Returns the dict this script's own `main` writes
    verbatim to CACHE_PATH (see shortstop.calvin.static_targets)."""
    raw_env = env.env
    p = raw_env.p
    cid = raw_env.cid
    scene = raw_env.scene

    snapshot = snapshot_env(env)
    try:
        targets = {}
        generated_from = {}
        endpoints = {}

        def _store_swept(cache_key, uid, joint_index):
            box_min, box_max, ll, ul, box_at_ll, box_at_ul = _swept_box(p, cid, uid, joint_index, cache_key)
            targets[cache_key] = {"min": box_min.tolist(), "max": box_max.tolist()}
            generated_from[cache_key] = {"ll": ll, "ul": ul}
            endpoints[cache_key] = {
                "ll": {"min": box_at_ll[0].tolist(), "max": box_at_ll[1].tolist()},
                "ul": {"min": box_at_ul[0].tolist(), "max": box_at_ul[1].tolist()},
            }
            print(
                f"  {cache_key}: ll={ll:.4f} ul={ul:.4f} -> "
                f"box min={np.round(box_min, 4).tolist()} max={np.round(box_max, 4).tolist()}"
            )

        for door in scene.doors:
            if door.name in DOOR_JOINT_NAMES:
                _store_swept(door.name, door.uid, door.joint_index)

        def _actuator_for_light(light_name):
            for obj in itertools.chain(scene.buttons, scene.switches):
                if obj.effect == light_name:
                    return obj
            raise ValueError(f"no button/switch found with effect=={light_name!r}")

        for light_name in LIGHT_NAMES:
            actuator = _actuator_for_light(light_name)
            # Cached under the light's name (the task config names the light).
            _store_swept(light_name, actuator.uid, actuator.joint_index)

        # plank_link: base__plank is a FIXED joint (calvin_table_D.urdf)
        # -- 1 real box, no sweep, no ll/ul at all.
        info = env.get_info()
        fixed_objects = info["scene_info"]["fixed_objects"]
        table_uid = fixed_objects["table"]["uid"]
        plank_link_id = fixed_objects["table"]["links"]["plank_link"]
        plank_min, plank_max = _real_aabb(p, cid, table_uid, plank_link_id)
        targets["plank_link"] = {"min": plank_min.tolist(), "max": plank_max.tolist()}
        generated_from["plank_link"] = {"fixed": True}
        print(f"  plank_link: fixed -> box min={np.round(plank_min, 4).tolist()} max={np.round(plank_max, 4).tolist()}")
    finally:
        restore_env_snapshot(env, snapshot)

    return {
        "scene": "calvin_scene_D",
        "generated_from": generated_from,
        "targets_world_frame": targets,
        # Boxes at both travel ends (none for the fixed plank_link).
        "targets_world_frame_endpoints": endpoints,
    }


@hydra.main(config_path="../mdt_policy/conf", config_name="mdt_evaluate")
def main(cfg):
    seed_everything(0, workers=True)
    env, policy, task_oracle, val_annotations, lang_embeddings = setup_env_and_policy(cfg, N_CANDIDATES)
    del policy, task_oracle, val_annotations, lang_embeddings  # only `env` is needed for this script

    # Any real initial_state loads the real scene/table/door/switch
    # bodies into PyBullet -- this script doesn't care WHICH one, since
    # every target it sweeps is static furniture, unaffected by which
    # sequence/subtask a real run would have picked.
    eval_sequences = get_sequences(2)
    initial_state, _ = eval_sequences[0]
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    print("[precompute] sweeping static (non-block) target boxes...")
    data = compute_static_target_sweeps(env)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[precompute] wrote {CACHE_PATH}")


if __name__ == "__main__":
    main()
