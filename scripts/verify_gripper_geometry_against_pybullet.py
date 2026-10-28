"""Check robot_geometry's gripper geometry against CALVIN's PyBullet Panda
along real rollouts (with a spread of gripper widths): each finger's joint
anchor (hand yaw, finger z offset, half-width spread) and the centered TCP
(gripper_tip_position). The URDF has no per-finger tip link, so the lateral
position of the finger tips is not checked. Link indices are looked up by
name.

Run from WSL2, inside `mdt_env`:

    cd SHORTSTOP
    python scripts/verify_gripper_geometry_against_pybullet.py
"""
import sys
from datetime import datetime
from pathlib import Path

import calvin_env
import hydra
import numpy as np
from pytorch_lightning import seed_everything

REPO_ROOT = Path(__file__).resolve().parents[1]
MDT_POLICY_ROOT = REPO_ROOT / "mdt_policy"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(MDT_POLICY_ROOT))

from mdt.evaluation.multistep_sequences import get_sequences  # noqa: E402
from mdt.evaluation.utils import get_default_beso_and_env, get_env_state_for_initial_condition  # noqa: E402
from mdt.utils.utils import get_last_checkpoint  # noqa: E402

from shortstop.calvin.config import N_CANDIDATES, REPLAN_STEPS  # noqa: E402
from shortstop.calvin.experiment import (  # noqa: E402
    _base_transform_from_env, _gripper_width_from_obs, _joint_angles_from_obs, _lang_goal, _to_action_tensor,
)
from shortstop.calvin.mdt_policy_client import ForwardOnlyPolicy  # noqa: E402
from shortstop.calvin.robot_geometry import finger_tip_capsules, gripper_tip_position, to_world_frame  # noqa: E402

N_SEQUENCES_TO_SAMPLE = 10  # a small spread of real subtask attempts is enough for a geometry check
STEPS_PER_SUBTASK = 100  # run well past a single replan window -- long enough to see the gripper actually close/open

RUN_OUTPUT_DIR = REPO_ROOT / "outputs" / "verify_gripper_geometry_runs" / f"run_{datetime.now():%Y%m%d_%H%M%S}"

_LOG_FILE = None


def _log(msg):
    print(msg)
    if _LOG_FILE is not None:
        _LOG_FILE.write(msg + "\n")
        _LOG_FILE.flush()


def _find_link_index(p, robot_uid, cid, child_link_name):
    """PyBullet has no by-name link lookup -- scan every joint's own
    childLinkName (getJointInfo index 12) and return the joint/link index
    whose child matches. A joint's own index IS its child link's index
    (pybullet's 1:1 joint<->child-link convention), so this also gives the
    link index to pass to getLinkState. Not hardcoded as a magic number so
    this stays correct even if the URDF's joint ordering changes upstream."""
    for i in range(p.getNumJoints(robot_uid, physicsClientId=cid)):
        info = p.getJointInfo(robot_uid, i, physicsClientId=cid)
        if info[12].decode("utf-8") == child_link_name:
            return i
    raise ValueError(f"no joint with child link {child_link_name!r} found on body {robot_uid}")


def _real_link_frame_position(p, cid, robot_uid, link_index):
    """worldLinkFramePosition (index 4, requires computeForwardKinematics)
    -- the actual URDF joint/link frame, NOT getLinkState(...)[0]'s
    center-of-mass frame (see scripts/verify_robot_geometry_against_
    pybullet.py's own docstring for why index 0 is wrong here -- the same
    gotcha applies: panda_leftfinger/panda_rightfinger both have a nonzero
    <inertial><origin>, confirmed from panda.urdf, so COM != joint frame
    for these links too)."""
    return np.array(p.getLinkState(robot_uid, link_index, computeForwardKinematics=1, physicsClientId=cid)[4])


@hydra.main(config_path="../mdt_policy/conf", config_name="mdt_evaluate")
def main(cfg):
    global _LOG_FILE
    RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _LOG_FILE = open(RUN_OUTPUT_DIR / "run.log", "w", encoding="utf-8")
    _log(f"[run] writing log to: {RUN_OUTPUT_DIR}")

    seed_everything(0, workers=True)

    checkpoint = get_last_checkpoint(Path(cfg.train_folder))
    model, env, _, lang_embeddings = get_default_beso_and_env(
        cfg.train_folder, cfg.dataset_path, checkpoint,
        eval_cfg_overwrite=cfg.eval_cfg_overwrite, device_id=cfg.device,
    )
    model.num_sampling_steps = cfg.num_sampling_steps
    model.sampler_type = cfg.sampler_type
    model.eval()

    task_oracle = hydra.utils.instantiate(cfg.tasks)
    val_annotations = cfg.annotations
    policy = ForwardOnlyPolicy(model, n_candidates=N_CANDIDATES)

    p, cid, robot_uid = env.env.p, env.env.cid, env.env.robot.robot_uid
    left_link = _find_link_index(p, robot_uid, cid, "panda_leftfinger")
    right_link = _find_link_index(p, robot_uid, cid, "panda_rightfinger")
    tcp_link = _find_link_index(p, robot_uid, cid, "tcp")
    _log(f"[setup] discovered link indices: panda_leftfinger={left_link} panda_rightfinger={right_link} tcp={tcp_link}")

    eval_sequences = get_sequences(2 * N_SEQUENCES_TO_SAMPLE)[0:N_SEQUENCES_TO_SAMPLE]

    left_errors, right_errors, centerline_errors, widths_seen = [], [], [], []
    for idx, (initial_state, eval_sequence) in enumerate(eval_sequences):
        seed_everything(1000 + idx, workers=True)
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

        subtask = eval_sequence[0]
        obs = env.get_obs()
        goal = _lang_goal(lang_embeddings, val_annotations, subtask)

        steps_done = 0
        while steps_done < STEPS_PER_SUBTASK:
            candidates = policy.propose({**obs, "goal": goal})
            chunk = candidates[0]
            for action_row in chunk[:REPLAN_STEPS]:
                obs, _, _, _ = env.step(_to_action_tensor(action_row))
                steps_done += 1

                joint_angles = _joint_angles_from_obs(obs)
                gripper_width = _gripper_width_from_obs(obs)
                widths_seen.append(gripper_width)

                # finger_tip_capsules()/gripper_tip_position() are LOCAL-
                # frame (see robot_geometry.py's module docstring), but
                # getLinkState reports true WORLD coordinates -- the same
                # base-frame conversion needed for g(a) (calvin/progress.py)
                # and the DH verify script; must transform before diffing
                # or the real (nonzero) robot_base_position dominates every
                # number here.
                base_position, base_orientation = _base_transform_from_env(env)

                (left_a_local, _), (right_a_local, _) = finger_tip_capsules(joint_angles, gripper_width)
                left_a = to_world_frame(left_a_local, base_position, base_orientation)
                right_a = to_world_frame(right_a_local, base_position, base_orientation)
                left_errors.append(float(np.linalg.norm(left_a - _real_link_frame_position(p, cid, robot_uid, left_link))))
                right_errors.append(float(np.linalg.norm(right_a - _real_link_frame_position(p, cid, robot_uid, right_link))))

                centerline_local = gripper_tip_position(joint_angles)
                centerline = to_world_frame(centerline_local, base_position, base_orientation)
                centerline_errors.append(float(np.linalg.norm(centerline - _real_link_frame_position(p, cid, robot_uid, tcp_link))))

                if steps_done >= STEPS_PER_SUBTASK:
                    break

    def _report(name, values):
        values = np.asarray(values)
        _log(f"  {name}: n={len(values)} mean={values.mean():.5f} max={values.max():.5f}")

    _log("[result] gripper_width actually sampled (real gripper_opening_width, meters):")
    widths = np.asarray(widths_seen)
    _log(f"  n={len(widths)} min={widths.min():.5f} max={widths.max():.5f} mean={widths.mean():.5f}")
    if widths.max() - widths.min() < 0.005:
        _log("  [warning] gripper barely moved across every sampled subtask -- the near-point check below "
             "only exercised a narrow slice of finger_tip_capsules()'s gripper_width range, not the full "
             "[0, 0.04] it needs to handle. Consider sampling different subtasks (grasping ones) if this "
             "warning fires.")

    _log("[result] per-finger near-point discrepancy (meters) between finger_tip_capsules() and real PyBullet state:")
    _report("left finger (near)", left_errors)
    _report("right finger (near)", right_errors)
    _log("[result] centerline far-point discrepancy (meters) between gripper_tip_position() and real PyBullet tcp link:")
    _report("centerline (far, no lateral offset)", centerline_errors)

    _log(
        "[run] DONE -- judge against the printed numbers (no hardcoded pass/fail threshold here). Near-point "
        "and centerline-far-point results, if within a few mm, confirm HAND_YAW_OFFSET/FINGER_JOINT_Z_OFFSET/"
        "GRIPPER_TIP_OFFSET and the gripper_width/2 lateral math. This script does NOT verify the far point's "
        "per-finger lateral placement (no per-finger tip link exists in panda.urdf to check it against) -- "
        "that remains a documented, unverified-by-construction assumption in finger_tip_capsules()."
    )
    _LOG_FILE.close()


if __name__ == "__main__":
    main()
