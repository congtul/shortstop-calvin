"""Shared plumbing for the CALVIN run scripts: env/policy setup, logging and
the pooled metrics over rollout results.

`sequence_results` below is a list (one per rollout) of the per-subtask
result lists the harnesses in shortstop.calvin.experiment return. Rates are
pooled over every decision, not averaged per subtask.
"""
import itertools
import json
from datetime import datetime
from pathlib import Path

import numpy as np


def setup_env_and_policy(cfg, n_candidates):
    """Load the MDT checkpoint, the env and the language embeddings from the
    mdt_evaluate hydra config; returns (env, policy, task_oracle,
    val_annotations, lang_embeddings). `policy.propose` returns
    `n_candidates` chunks."""
    import hydra
    from mdt.evaluation.utils import get_default_beso_and_env
    from mdt.utils.utils import get_last_checkpoint
    from shortstop.calvin.mdt_policy_client import ForwardOnlyPolicy

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
    policy = ForwardOnlyPolicy(model, n_candidates=n_candidates)
    return env, policy, task_oracle, cfg.annotations, lang_embeddings


def _attempts(sequence_results):
    return [a for attempts in sequence_results for a in attempts]


def _summary(values, percentiles=(10, 90)):
    values = np.asarray(values, dtype=float)
    stats = {"n": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
             "min": float(values.min()), "max": float(values.max())}
    stats.update({f"p{p}": float(np.percentile(values, p)) for p in percentiles})
    return stats


def clearance_stats(sequence_results):
    """Distribution of each attempt's min clearance (attempts with an obstacle)."""
    clearances = [a["min_clearance"] for a in _attempts(sequence_results) if a["min_clearance"] is not None]
    return _summary(clearances) if clearances else None


def _pooled_ratio(sequence_results, numerator):
    attempts = _attempts(sequence_results)
    n_decisions = sum(a["n_decisions"] for a in attempts)
    return None if n_decisions == 0 else sum(a.get(numerator, 0) for a in attempts) / n_decisions


def shield_activation_rate(sequence_results):
    """Decisions with any rejected candidate / all decisions."""
    return _pooled_ratio(sequence_results, "n_activated")


def fallback_rate(sequence_results):
    """Decisions with no admissible candidate / all decisions."""
    return _pooled_ratio(sequence_results, "n_fallback")


def _latency(sequence_results, field):
    values = [t for a in _attempts(sequence_results) for t in a.get(field, [])]
    return _summary(values, percentiles=(95,)) if values else None


def latency_stats(sequence_results):
    """shield.select() time per decision (ms)."""
    return _latency(sequence_results, "latencies_ms")


def propose_latency_stats(sequence_results):
    """policy.propose() time per decision (ms)."""
    return _latency(sequence_results, "propose_latencies_ms")


def progress_latency_stats(sequence_results):
    """g(a) time per decision (ms); shielded runs only."""
    return _latency(sequence_results, "progress_latencies_ms")


def intervention_precision(sequence_results):
    """Rejected candidates that really collide in the simulator / all rejected."""
    attempts = _attempts(sequence_results)
    total = sum(a["rejected_total"] for a in attempts)
    return None if total == 0 else sum(a["rejected_truly_unsafe"] for a in attempts) / total


def make_run_output_dir(repo_root, run_name, tuning_mode):
    return repo_root / "outputs" / run_name / f"run_{datetime.now():%Y%m%d_%H%M%S}_{'tuning' if tuning_mode else 'eval'}"


def make_logger(log_file_path):
    """`log(msg)` that prints and appends to `log_file_path` (`log.file`
    is the open handle)."""
    log_file = open(log_file_path, "w", encoding="utf-8")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log.file = log_file
    return log


def write_results_json(results_path, payload):
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# --- Baselines on the frozen scenes -----------------------------------------

def _product(**axes):
    keys = list(axes)
    return [dict(zip(keys, values)) for values in itertools.product(*axes.values())]


# Val tuning grids. Conf-Thresh's thresholds span the measured candidate TCP
# disagreement on val (K = 8, 8 rows: p50 1.6 mm, p90 4.6 mm, p99 9.1 mm; final K = 4,
# H = 4: p50 0.7 mm, p90 2.1 mm, p99 6.7 mm); MPC-Filter has
# nothing to tune.
GRIDS = {
    "stl_monitor": _product(epsilon=[0.0, 0.01, 0.02]),
    "mpc_filter": [{}],
    "cbf": _product(alpha=[0.25, 0.5]),
    "conf_thresh": _product(threshold=[0.002, 0.003, 0.005]),
    "shortstop": _product(epsilon=[0.0, 0.01, 0.02], trust_region=[0.03, 0.05, 0.1], step_size=[0.02, 0.05]),
}


def make_shield(baseline, params):
    """The shield for `baseline` with `params`; None for "unshielded"."""
    from shortstop.calvin.config import HORIZON_STEPS, MODEL_ERROR, REPLAN_STEPS, W_BAR
    from shortstop.calvin.shield import (
        ArmCBFShield, ArmConfThreshShield, ArmMPCFilterShield, ArmReachOnlyShield, ArmRepairShield,
        ArmSTLMonitorShield,
    )
    if baseline == "unshielded":
        return None
    if baseline == "reach_only":  # ablation: ShortStop's certify (same tube, margin, joint limits), no repair
        return ArmReachOnlyShield(obstacles=[], w_bar=W_BAR, model_error=MODEL_ERROR, horizon_steps=HORIZON_STEPS)
    if baseline == "best_of_n":
        return ArmConfThreshShield(disagreement_threshold=float("inf"), horizon_steps=HORIZON_STEPS)
    if baseline == "stl_monitor":
        return ArmSTLMonitorShield(obstacles=[], epsilon=params["epsilon"], horizon_steps=HORIZON_STEPS)
    if baseline == "mpc_filter":
        return ArmMPCFilterShield(obstacles=[], w_bar=W_BAR, model_error=MODEL_ERROR, horizon_steps=HORIZON_STEPS)
    if baseline == "cbf":
        return ArmCBFShield(obstacles=[], w_bar=W_BAR, model_error=MODEL_ERROR, alpha=params["alpha"],
                            steps=REPLAN_STEPS)
    if baseline == "conf_thresh":
        return ArmConfThreshShield(disagreement_threshold=params["threshold"], horizon_steps=HORIZON_STEPS)
    if baseline == "shortstop":
        return ArmRepairShield(**{"obstacles": [], "w_bar": W_BAR, "model_error": MODEL_ERROR,
                                  "horizon_steps": HORIZON_STEPS, **params})
    raise ValueError(f"unknown baseline {baseline!r}")


def warm_up(baseline, params):
    """One select() of a fresh `baseline` shield on a synthetic blocked chunk,
    so one-off costs (cvxpy canonicalization, solver loading, caches) do not
    land in the first measured decision; then collect and freeze the heap
    (model, env, hydra) so a full GC pass (~2.5 s here) never lands in a
    timed select."""
    import gc

    _warm_up_select(baseline, params)
    gc.collect()
    gc.freeze()


def _warm_up_select(baseline, params):
    from shortstop.calvin.config import CHUNK_LENGTH, N_CANDIDATES
    from shortstop.calvin.reach import CALVIN_ACTION_SCALE
    from shortstop.calvin.robot_geometry import gripper_tip_position
    from shortstop.calvin.scene_manifest import START_JOINT_ANGLES
    from shortstop.env import Obstacle

    shield = make_shield(baseline, params)
    if shield is None:
        return
    q = np.asarray(START_JOINT_ANGLES, dtype=float)
    if hasattr(shield, "obstacles"):
        shield.obstacles = [Obstacle(gripper_tip_position(q) + np.array([0.12, 0.0, 0.0]), 0.05)]
    chunk = np.zeros((CHUNK_LENGTH, 7))
    chunk[:, 0] = 0.015 / CALVIN_ACTION_SCALE
    shield.select(q, [chunk.copy() for _ in range(N_CANDIDATES)], [0.0] * N_CANDIDATES)


def run_scene(env, policy, task_oracle, val_annotations, lang_embeddings, get_env_state_for_initial_condition,
              scene, seed_k, shield, ep_len, fallback="hold", retreat_step=None, measure_intervention_precision=False,
              log_soundness=False, delay_steps=0, delay_aware=False, use_progress=True):
    """One rollout of a frozen scene (policy seed = scene seed + 1000 seed_k),
    unshielded when `shield` is None; `use_progress=False` scores every
    candidate 0 (no g(a), ablation). Returns a JSON-able result row."""
    from pytorch_lightning import seed_everything
    from shortstop.calvin.config import FALLBACK_RETREAT_STEP, HORIZON_STEPS, REPLAN_STEPS
    from shortstop.calvin.experiment import run_calvin_shielded_sequence, run_calvin_unshielded_sequence

    seed_everything(scene["policy_seed"] + 1000 * seed_k, workers=True)
    common = (env, policy, task_oracle, lang_embeddings, scene["initial_state"], scene["eval_sequence"],
              val_annotations, get_env_state_for_initial_condition)
    if shield is None:
        a = run_calvin_unshielded_sequence(*common, ep_len=ep_len, replan_steps=REPLAN_STEPS,
                                           obstacles=scene["obstacles"], delay_steps=delay_steps)[0]
    else:
        if hasattr(shield, "obstacles"):
            shield.obstacles = []  # the harness sets the scene's obstacles
        a = run_calvin_shielded_sequence(
            *common, shield, ep_len=ep_len, replan_steps=REPLAN_STEPS, horizon_steps=HORIZON_STEPS,
            obstacles=scene["obstacles"], fallback_controller=fallback,
            retreat_step=FALLBACK_RETREAT_STEP if retreat_step is None else retreat_step,
            measure_intervention_precision=measure_intervention_precision, log_soundness=log_soundness,
            delay_steps=delay_steps, delay_aware=delay_aware, use_progress=use_progress,
        )[0]
    row = {
        "scene_id": scene["scene_id"], "task_name": scene["task_name"], "seed_k": seed_k,
        "policy_seed": scene["policy_seed"] + 1000 * seed_k,
        "violated": bool(a["violated"]), "reached": bool(a["reached"]), "steps_taken": int(a["steps_taken"]),
        "min_clearance": a["min_clearance"],
        "propose_ms": [round(t, 2) for t in a["propose_latencies_ms"]],
    }
    if shield is not None:
        row.update({k: a[k] for k in ("n_decisions", "n_activated", "n_fallback", "rejected_total",
                                      "rejected_truly_unsafe", "g_a_phase2_step", "status_counts",
                                      "decision_fallback")})
        row["select_ms"] = [round(t, 2) for t in a["latencies_ms"]]
        row["progress_ms"] = [round(t, 2) for t in a["progress_latencies_ms"]]
        if "soundness" in a:
            row["soundness"] = [[d, k, round(p, 5), round(t, 5), round(e, 5)] for d, k, p, t, e in a["soundness"]]
        if "disagreement" in a:  # Conf-Thresh: each candidate's predicted TCP distance to the K-centroid (m)
            row["disagreement"] = [[round(x, 6) for x in d] for d in a["disagreement"]]
    return row


def rollout_metrics(rows, unshielded_rows=None):
    """Paper metrics over result rows (one rollout = one slot): safe success,
    violation, timeout, shield activation and fallback rates (pooled over
    decisions), intervention precision, recovery rate, latencies (ms) and,
    given the unshielded rows of the same scenes/seeds, conservatism cost."""
    from shortstop.calvin.metrics import conservatism_cost, recovery_rate

    n = len(rows)
    out = {
        "n": n,
        "safe_success": float(np.mean([r["reached"] for r in rows])),
        "violation": float(np.mean([r["violated"] for r in rows])),
        "timeout": float(np.mean([not r["reached"] and not r["violated"] for r in rows])),
        "mean_steps": float(np.mean([r["steps_taken"] for r in rows])),
    }
    if rows and "n_decisions" in rows[0]:
        results = [[r] for r in rows]
        out.update({
            "status_counts": _sum_counts(r.get("status_counts", {}) for r in rows),
            "activation": shield_activation_rate(results),
            "fallback": fallback_rate(results),
            "intervention_precision": intervention_precision(results),
            "recovery": recovery_rate(results),
            "select_ms": _latency([[{"latencies_ms": r["select_ms"]} for r in rows]], "latencies_ms"),
            # Paper latency: shield time added per decision = select + g(a) for the
            # shields that rank candidates by g(a); MPC-Filter/CBF do not use it.
            "added_ms": _latency([[{"latencies_ms": [
                s + (0.0 if r.get("baseline") in ("mpc_filter", "cbf") else g)
                for s, g in zip(r["select_ms"], r["progress_ms"])]} for r in rows]], "latencies_ms"),
            "progress_ms": _latency([[{"progress_latencies_ms": r["progress_ms"]} for r in rows]],
                                    "progress_latencies_ms"),
        })
    out["propose_ms"] = _latency([[{"propose_latencies_ms": r["propose_ms"]} for r in rows]], "propose_latencies_ms")
    if unshielded_rows is not None:
        paired = {(r["scene_id"], r["seed_k"]): r for r in unshielded_rows}
        keys = [(r["scene_id"], r["seed_k"]) for r in rows if (r["scene_id"], r["seed_k"]) in paired]
        mine = {(r["scene_id"], r["seed_k"]): r for r in rows}
        out["conservatism_cost"] = conservatism_cost([[paired[k]] for k in keys], [[mine[k]] for k in keys])
    return out


def _sum_counts(dicts):
    total = {}
    for d in dicts:
        for key, value in d.items():
            total[key] = total.get(key, 0) + value
    return total


def bootstrap(rows, field, n_resamples=10_000, seed=0):
    """(mean, 2.5th, 97.5th percentile) of mean(row[field]) over rollouts
    resampled with replacement."""
    values = np.asarray([float(r[field]) for r in rows])
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n_resamples, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
