"""Check robot_geometry.panda_frames (modified-DH FK) against CALVIN's
PyBullet Panda on configurations from real rollouts, per frame: base from
the base pose, joints 1-7 and the flange from the URDF link frames
(PyBullet's default link position is the center of mass, not the frame).

Run from WSL2, inside `mdt_env`:

    cd SHORTSTOP
    python scripts/verify_robot_geometry_against_pybullet.py
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
    _base_transform_from_env, _joint_angles_from_obs, _lang_goal, _to_action_tensor,
)
from shortstop.calvin.mdt_policy_client import ForwardOnlyPolicy  # noqa: E402
from shortstop.calvin.robot_geometry import N_JOINTS, panda_frames, to_world_frame  # noqa: E402

N_SEQUENCES_TO_SAMPLE = 10  # a small spread of real configs is enough for a geometry check, not a full sweep

RUN_OUTPUT_DIR = REPO_ROOT / "outputs" / "verify_robot_geometry_runs" / f"run_{datetime.now():%Y%m%d_%H%M%S}"

_LOG_FILE = None


def _log(msg):
    print(msg)
    if _LOG_FILE is not None:
        _LOG_FILE.write(msg + "\n")
        _LOG_FILE.flush()


def _pybullet_frames(env, joint_angles):
    """The same 9 positions panda_frames(joint_angles) predicts, read
    directly from PyBullet's own simulated robot at whatever its REAL
    current joint state happens to be (the caller is responsible for
    having already driven the sim to `joint_angles` via env.step() --
    this function only reads, it doesn't set anything).

    NOT `getLinkState(...)[0]`/`getBasePositionAndOrientation(...)[0]` --
    those report each link's CENTER-OF-MASS frame, not the URDF joint/
    link frame panda_frames()'s DH chain represents (confirmed: PyBullet's
    own quickstart guide -- "position and orientation ... given in the
    center of mass frame"; also github.com/bulletphysics/bullet3/issues/
    2096, "get base URDF link frame position problem"). The two frames
    coincide only when a link's <inertial><origin> offset is zero -- true
    for a massless/fixed frame like the flange, false for a real link
    with its own motor-housing mass distribution (elbow/forearm/etc),
    which is exactly the pattern the first real run showed here: joints
    1-7 off by 4-13cm (each link's own fixed inertial offset, rotated
    into world -- constant across configs since rotation preserves the
    offset vector's length, matching the near-zero mean/max spread
    observed), base off by a constant 5cm, but flange within ~1mm.

    Fix: for links 0-6 (the 7 joints), pass `computeForwardKinematics=1`
    and read index 4 (`worldLinkFramePosition`, the actual URDF link
    frame) instead of index 0. The base has no link index to query this
    way (`getLinkState` doesn't accept -1) -- recover its URDF frame from
    its center-of-mass pose via `getDynamicsInfo(robot_uid, -1)`'s own
    local_inertial_pos/orn (indices 3/4), inverted and composed with the
    COM pose (`invertTransform`/`multiplyTransforms`), the standard
    PyBullet recipe for this exact COM<->link-frame conversion."""
    p, cid, robot = env.env.p, env.env.cid, env.env.robot

    base_com_pos, base_com_orn = p.getBasePositionAndOrientation(robot.robot_uid, physicsClientId=cid)
    local_inertial_pos, local_inertial_orn = p.getDynamicsInfo(robot.robot_uid, -1, physicsClientId=cid)[3:5]
    inv_inertial_pos, inv_inertial_orn = p.invertTransform(local_inertial_pos, local_inertial_orn)
    base_link_pos, _ = p.multiplyTransforms(base_com_pos, base_com_orn, inv_inertial_pos, inv_inertial_orn)
    base = np.array(base_link_pos)

    joints = [
        np.array(p.getLinkState(robot.robot_uid, i, computeForwardKinematics=1, physicsClientId=cid)[4])
        for i in range(N_JOINTS)
    ]
    flange = np.array(p.getLinkState(
        robot.robot_uid, robot.end_effector_link_id, computeForwardKinematics=1, physicsClientId=cid,
    )[4])
    return np.stack([base] + joints + [flange])


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

    eval_sequences = get_sequences(2 * N_SEQUENCES_TO_SAMPLE)[0:N_SEQUENCES_TO_SAMPLE]

    per_frame_errors = [[] for _ in range(N_JOINTS + 2)]
    for idx, (initial_state, eval_sequence) in enumerate(eval_sequences):
        seed_everything(1000 + idx, workers=True)
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

        for subtask in eval_sequence[:1]:  # one subtask per sequence is enough spread for a geometry check
            obs = env.get_obs()
            goal = _lang_goal(lang_embeddings, val_annotations, subtask)
            for _ in range(REPLAN_STEPS):
                candidates = policy.propose({**obs, "goal": goal})
                chunk = candidates[0]
                for action_row in chunk[:REPLAN_STEPS]:
                    obs, _, _, _ = env.step(_to_action_tensor(action_row))
                    joint_angles = _joint_angles_from_obs(obs)
                    # panda_frames() is expressed in the robot's own LOCAL
                    # base frame (see robot_geometry.py's module docstring
                    # -- _panda_transforms starts at T=eye(4), no base
                    # offset folded in); PyBullet reports world coordinates.
                    base_position, base_orientation = _base_transform_from_env(env)
                    ours_local = panda_frames(joint_angles)
                    ours = np.stack([
                        to_world_frame(point, base_position, base_orientation) for point in ours_local
                    ])
                    real = _pybullet_frames(env, joint_angles)
                    for i in range(N_JOINTS + 2):
                        per_frame_errors[i].append(float(np.linalg.norm(ours[i] - real[i])))
                break  # one replan cycle per subtask is plenty of samples

    _log("[result] per-frame discrepancy (meters) between panda_frames() and real PyBullet state:")
    for i, errors in enumerate(per_frame_errors):
        values = np.asarray(errors)
        label = "base" if i == 0 else ("flange" if i == N_JOINTS + 1 else f"joint{i}")
        _log(f"  frame {i} ({label}): n={len(values)} mean={values.mean():.5f} max={values.max():.5f}")

    _log(
        "[run] DONE -- a frame off by more than a few mm means PANDA_DH/FLANGE_OFFSET need correcting."
    )
    _LOG_FILE.close()


if __name__ == "__main__":
    main()
