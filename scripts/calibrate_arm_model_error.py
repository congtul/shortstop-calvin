"""Measure the shields' model error on real CALVIN rollouts.

Runs the unshielded policy without obstacles, executing each chunk's first
HORIZON_STEPS rows open loop, and records for k = 1..H how far the
TCP-Jacobian prediction of the arm frames and TCP (and of the joints) after
k rows lands from where CALVIN actually put the arm. The tube radius at step
k is model_error[k-1] = p99 x 1.25 of the k-step residual (the paper's Table
VII recipe, per horizon step, so the tube contains the real arm at every
step as Assumption 1 requires). Every third decision also holds the arm
for FREEZE_PROBE_STEPS rows and records how far the arm still moves,
cycling the freeze fallback (zero actions), a
brake (IK target moved onto the current TCP pose) and a joint hold
(angles of the first hold step). FREEZE_PROBE_STEPS (env var, default 8). Both predictors are
measured: position columns only, and position + orientation
(`reach.nominal_joint_trajectory` with the TCP orientation). Writes calibration.json and samples.npz (every raw sample) in the run directory; copy
model_error into shortstop/calvin/config.MODEL_ERROR (STL-Monitor keeps 0).

Run from WSL2, inside `mdt_env`:

    cd SHORTSTOP
    python scripts/calibrate_arm_model_error.py
"""
import json
import os
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

from shortstop.calvin.reach import CALVIN_ACTION_SCALE, CALVIN_ORIENTATION_SCALE, nominal_joint_trajectory  # noqa: E402
from shortstop.calvin.config import HORIZON_STEPS, N_CANDIDATES  # noqa: E402
from shortstop.calvin.experiment import (  # noqa: E402
    _brake_action, _gripper_action_from_obs, _hold_step, _joint_angles_from_obs, _lang_goal, _robot_obs_raw,
    _to_action_tensor,
)
from shortstop.calvin.robot_geometry import gripper_tip_position, panda_frames  # noqa: E402
from shortstop.calvin.mdt_policy_client import ForwardOnlyPolicy  # noqa: E402

# Calibration recipe: per-step p99 of the prediction error, inflated by 1.25.
QUANTILE = 0.99
SAFETY_FACTOR = 1.25

RUN_OUTPUT_DIR = REPO_ROOT / "outputs" / "calibrate_arm_model_error_runs" / f"run_{datetime.now():%Y%m%d_%H%M%S}"

_LOG_FILE = None


def _log(msg):
    print(msg)
    if _LOG_FILE is not None:
        _LOG_FILE.write(msg + "\n")
        _LOG_FILE.flush()


def _residual_percentiles(residuals):
    values = np.asarray(residuals)
    return {
        "n": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)), "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)), "min": float(values.min()), "max": float(values.max()),
    }


FREEZE_PROBE_STEPS = int(os.environ.get("FREEZE_PROBE_STEPS", 8))
PROBE_MODES = ("freeze", "brake", "hold")
MODELS = ("position", "pose")  # predictor: position columns only / position + orientation


def _points(joint_angles):
    return np.vstack([panda_frames(joint_angles), gripper_tip_position(joint_angles)])


def _run_subtask(env, policy, task_oracle, lang_embeddings, subtask, val_annotations, ep_len, out):
    """Unshielded P-execute loop (no obstacle), HORIZON_STEPS rows per
    decision, appending k-step residuals of both predictors to
    out[model]["frame"/"joint"][k-1]; every third decision also holds the arm
    (alternately "freeze" = zero actions, "brake" = target onto the current
    pose, then zeros) and appends the motion j rows in to out[mode][j-1]."""
    obs = env.get_obs()
    goal = _lang_goal(lang_embeddings, val_annotations, subtask)
    start_info = env.get_info()
    steps_taken, decision = 0, 0
    while steps_taken < ep_len:
        q0 = _joint_angles_from_obs(obs)
        chunk = policy.propose({**obs, "goal": goal})[0]
        predicted = {"position": nominal_joint_trajectory(q0, chunk[:HORIZON_STEPS]),
                     "pose": nominal_joint_trajectory(q0, chunk[:HORIZON_STEPS], _robot_obs_raw(obs)[3:6])}
        for k, action_row in enumerate(chunk[:HORIZON_STEPS], 1):
            obs, _, _, current_info = env.step(_to_action_tensor(action_row))
            steps_taken += 1
            q = _joint_angles_from_obs(obs)
            for model in MODELS:
                qp = predicted[model][k]
                out[model]["frame"][k - 1].append(float(np.max(np.linalg.norm(_points(q) - _points(qp), axis=1))))
                out[model]["joint"][k - 1].append(float(np.max(np.abs(q - qp))))
            if task_oracle.get_task_info_for_set(start_info, current_info, {subtask}) or steps_taken >= ep_len:
                return
        decision += 1
        if decision % 3 == 0:
            mode = PROBE_MODES[(decision // 3) % len(PROBE_MODES)]
            hold = np.zeros(7)
            hold[6] = _gripper_action_from_obs(obs)
            first = hold.copy()
            if mode == "brake":
                first[:6] = _brake_action(env, obs)
            rows = [first] + [hold] * (FREEZE_PROBE_STEPS - 1)
            start = _points(_joint_angles_from_obs(obs))
            anchor = list(env.env.robot.get_observation()[1]["arm_joint_states"])
            for j, row in enumerate(rows, 1):
                if mode == "hold":
                    obs, _, _, _ = _hold_step(env, hold[6], anchor)
                else:
                    obs, _, _, _ = env.step(_to_action_tensor(row))
                steps_taken += 1
                moved = np.linalg.norm(_points(_joint_angles_from_obs(obs)) - start, axis=1)
                out[mode][j - 1].append(float(np.max(moved)))
                out[mode + "_tcp"][j - 1].append(float(moved[-1]))


@hydra.main(config_path="../mdt_policy/conf", config_name="mdt_evaluate")
def main(cfg):
    global _LOG_FILE
    RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _LOG_FILE = open(RUN_OUTPUT_DIR / "run.log", "w", encoding="utf-8")
    _log(f"[run] writing log to: {RUN_OUTPUT_DIR}")
    _log(f"[run] config: num_sequences={cfg.num_sequences} ep_len={cfg.ep_len} horizon_steps={HORIZON_STEPS} "
         f"sampler_type={cfg.sampler_type} num_sampling_steps={cfg.num_sampling_steps} n_candidates={N_CANDIDATES}")

    seed_everything(0, workers=True)

    checkpoint = get_last_checkpoint(Path(cfg.train_folder))
    model, env, _, lang_embeddings = get_default_beso_and_env(
        cfg.train_folder, cfg.dataset_path, checkpoint,
        eval_cfg_overwrite=cfg.eval_cfg_overwrite, device_id=cfg.device,
    )
    model.num_sampling_steps = cfg.num_sampling_steps
    model.sampler_type = cfg.sampler_type
    if cfg.sigma_min is not None:
        model.sigma_min = cfg.sigma_min
    if cfg.sigma_max is not None:
        model.sigma_max = cfg.sigma_max
    if cfg.noise_scheduler is not None:
        model.noise_scheduler = cfg.noise_scheduler
    model.eval()

    task_oracle = hydra.utils.instantiate(cfg.tasks)
    val_annotations = cfg.annotations
    policy = ForwardOnlyPolicy(model, n_candidates=N_CANDIDATES)

    # A measurement, not a tuned choice: CALVIN's first N chained sequences.
    N = cfg.num_sequences
    eval_sequences = get_sequences(2 * N)[0:N]
    SEQUENCE_SEED_BASE = 1000

    robot = env.env.robot
    orn_scale = robot.max_rel_orn * robot.magic_scaling_factor_orn
    assert abs(orn_scale - CALVIN_ORIENTATION_SCALE) < 1e-12 and abs(
        robot.max_rel_pos * robot.magic_scaling_factor_pos - CALVIN_ACTION_SCALE) < 1e-12, "action scale mismatch"
    _log(f"[env] max_rel_pos={robot.max_rel_pos} magic_pos={robot.magic_scaling_factor_pos} "
         f"max_rel_orn={robot.max_rel_orn} magic_orn={robot.magic_scaling_factor_orn} use_target_pose={robot.use_target_pose}")
    out = {model: {"frame": [[] for _ in range(HORIZON_STEPS)], "joint": [[] for _ in range(HORIZON_STEPS)]}
           for model in MODELS}
    out.update({mode: [[] for _ in range(FREEZE_PROBE_STEPS)] for m in PROBE_MODES for mode in (m, m + "_tcp")})
    for idx, (initial_state, eval_sequence) in enumerate(eval_sequences):
        seed_everything(SEQUENCE_SEED_BASE + idx, workers=True)
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        for subtask in eval_sequence:
            _run_subtask(env, policy, task_oracle, lang_embeddings, subtask, val_annotations, cfg.ep_len, out)
        _log(f"[progress] sequence {idx + 1}/{len(eval_sequences)}")

    result = {"quantile": QUANTILE, "safety_factor": SAFETY_FACTOR, "orientation_scale": orn_scale}
    for model in MODELS:
        result[model] = {}
        for key in ("frame", "joint"):
            result[model][key] = []
            for k, values in enumerate(out[model][key], 1):
                stats = _residual_percentiles(values)
                result[model][key].append(stats)
                _log(f"[{model} {key}] k={k} n={stats['n']} median={stats['median']:.5f} p99={stats['p99']:.5f} "
                     f"max={stats['max']:.5f}")
        result[model]["model_error"] = [round(st["p99"] * SAFETY_FACTOR, 5) for st in result[model]["frame"]]
        result[model]["joint_error"] = [round(st["p99"] * SAFETY_FACTOR, 5) for st in result[model]["joint"]]
        _log(f"[calibrated {model}] model_error per step = {result[model]['model_error']}")
        _log(f"[calibrated {model}] joint_error per step = {result[model]['joint_error']}")
    for mode in [m for p in PROBE_MODES for m in (p, p + "_tcp")]:
        result[mode] = [_residual_percentiles(v) for v in out[mode] if v]
        for j, st in enumerate(result[mode], 1):
            _log(f"[{mode}] j={j} n={st['n']} median={st['median']:.5f} p99={st['p99']:.5f} max={st['max']:.5f}")
    (RUN_OUTPUT_DIR / "calibration.json").write_text(json.dumps(result, indent=1))
    np.savez(RUN_OUTPUT_DIR / "samples.npz", **{f"{m}_{key}_k{k + 1}": np.asarray(v)
                                                for m in MODELS for key in ("frame", "joint")
                                                for k, v in enumerate(out[m][key])},
             **{f"{mode}_j{j + 1}": np.asarray(v) for p in PROBE_MODES for mode in (p, p + "_tcp")
                for j, v in enumerate(out[mode]) if v})
    _LOG_FILE.close()


if __name__ == "__main__":
    main()
