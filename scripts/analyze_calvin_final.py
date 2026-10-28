"""Main CALVIN run (paper Table 1 and Sect. 6.1-6.2): safe success /
violation with 95% bootstrap CIs, activation, fallback, added latency, paired
bootstrap tests vs ShortStop (same scene and seed) with Holm's correction,
conservatism cost, latency breakdown, collisions, tube soundness on the
soundness run, the statistical tube bound (Corollary 2) and scene coverage.

    python scripts/analyze_calvin_final.py
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from shortstop.calvin.config import HORIZON_STEPS, MODEL_ERROR  # noqa: E402
from shortstop.calvin.reach import model_error_at  # noqa: E402

FINAL = REPO_ROOT / "outputs" / "calvin_final"
CALIBRATION = REPO_ROOT / "outputs" / "calibrate_arm_model_error_runs" / "run_20261012_120501"
SCENES = REPO_ROOT / "shortstop" / "calvin" / "data" / "scenes" / "final_scenes.json"
METHODS = [("Unshielded", "unshielded"), ("Conf-Thresh", "conf_thresh_hold"), ("STL-Monitor", "stl_monitor_hold"),
           ("MPC-Filter", "mpc_filter_hold"), ("CBF", "cbf_hold"), ("ShortStop", "shortstop_hold")]
N_BOOT = 10000


def _load(name):
    return {(e["scene_id"], e["seed_k"]): e for e in map(json.loads, open(FINAL / f"{name}.jsonl"))}


def _safe(e):
    return e["reached"] and not e["violated"]


def _violated(e):
    return e["violated"]


def main():
    runs = {label: _load(name) for label, name in METHODS}
    keys = sorted(runs["ShortStop"])
    assert all(set(r) == set(keys) for r in runs.values()), "methods must share scenes and seeds"
    boot = np.random.default_rng(0).integers(0, len(keys), (N_BOOT, len(keys)))

    def values(label, fn):
        return np.array([fn(runs[label][k]) for k in keys], dtype=float)

    def ci(x):
        means = x[boot].mean(1)
        return f"{x.mean():.3f} [{np.percentile(means, 2.5):.3f}, {np.percentile(means, 97.5):.3f}]"

    print("| Method | Safe success [95% CI] | Violation [95% CI] | Activation | Fallback | Added ms median / p95 / p99 / max |")
    print("|---|---|---|---|---|---|")
    for label, name in METHODS:
        pooled = json.load(open(FINAL / f"{name}_summary.json"))["pooled"]
        ms = pooled.get("added_ms") or {}
        added = [m for e in runs[label].values() for m in _added_ms(label, e)]
        lat = (f"{np.median(added):.0f} / {np.percentile(added, 95):.0f} / {np.percentile(added, 99):.0f} / "
               f"{max(added):.0f}") if added else "-"
        assert not added or abs(np.median(added) - ms["median"]) < 1.0
        print(f"| {label} | {ci(values(label, _safe))} | {ci(values(label, _violated))} | "
              f"{pooled.get('activation') or 0:.3f} | {pooled.get('fallback') or 0:.3f} | {lat} |")

    print("\nPaired vs ShortStop (ShortStop - other, paired bootstrap, two-sided p; discordant scenes)")
    primary = []
    for label, _ in METHODS[:-1]:
        for metric, fn in (("safe success", _safe), ("violation", _violated)):
            a, b = values("ShortStop", fn), values(label, fn)
            d = (a - b)[boot].mean(1)
            p = _p_two_sided(d)
            primary.append((f"{label} {metric}", p))
            print(f"  {label:12s} {metric:12s} {(a - b).mean():+.3f} [{np.percentile(d, 2.5):+.3f}, "
                  f"{np.percentile(d, 97.5):+.3f}] p={p:.4f}  ShortStop-only {int(((a == 1) & (b == 0)).sum())}, "
                  f"other-only {int(((a == 0) & (b == 1)).sum())}")
    print("\nHolm's correction over these ten comparisons (adjusted p)")
    for (name, p), adj in zip(primary, _holm([p for _, p in primary])):
        print(f"  {name:25s} p={p:.4f} -> {adj:.4f}")

    collided = [k for k in keys if runs["Unshielded"][k]["violated"]]
    print(f"\nScenes where Unshielded collides: {len(collided)}; safe success there: "
          + ", ".join(f"{label} {np.mean([_safe(runs[label][k]) for k in collided]):.3f}"
                      for label in ("ShortStop", "CBF", "MPC-Filter")))

    _conservatism(runs, keys, boot)
    _latency(runs["ShortStop"])
    _collisions(runs)
    _soundness()
    _tube_bound()
    _coverage()


def _p_two_sided(d):
    return min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))


def _holm(p):
    """Holm step-down adjusted p-values, in the input order."""
    order = np.argsort(p)
    adjusted, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted


def _conservatism(runs, keys, boot):
    """Success lost vs Unshielded on the scenes where Unshielded does not collide,
    with scene-bootstrap CIs and paired tests vs ShortStop."""
    benign = np.array([not runs["Unshielded"][k]["violated"] for k in keys], dtype=float)
    reached = {label: np.array([runs[label][k]["reached"] for k in keys], dtype=float) for label, _ in METHODS}
    lost = {label: (reached["Unshielded"] - reached[label]) * benign for label, _ in METHODS[1:]}

    def boot_cost(x):
        return x[boot].sum(1) / benign[boot].sum(1)

    print(f"\nConservatism cost (points, {int(benign.sum())} scenes where Unshielded does not collide)")
    for label, _ in METHODS[1:]:
        c = boot_cost(lost[label])
        line = (f"  {label:12s} {100 * lost[label].sum() / benign.sum():.1f} "
                f"[{100 * np.percentile(c, 2.5):.1f}, {100 * np.percentile(c, 97.5):.1f}]")
        if label in ("MPC-Filter", "CBF"):
            d = lost["ShortStop"] - lost[label]
            line += f"  ShortStop - {label} {100 * d.sum() / benign.sum():+.1f} p={_p_two_sided(boot_cost(d)):.2f}"
        print(line)


def _latency(rows):
    """ShortStop per decision, warm-up decision excluded: policy sampling, g(a)
    and the full decision (sampling + selection/repair + g(a))."""
    decisions = [(p, s, g) for e in rows.values()
                 for p, s, g in list(zip(e["propose_ms"], e["select_ms"], e["progress_ms"]))[1:]]
    propose, select, progress = (np.array(x) for x in zip(*decisions))
    print(f"\nShortStop latency per decision (median ms): MDT sampling {np.median(propose):.0f}, "
          f"g(a) {np.median(progress):.0f}, full decision {np.median(propose + select + progress):.0f}; "
          f"longest stall {max(_longest_stall(e) for e in rows.values())} held steps")


def _longest_stall(e):
    best = run = 0
    for held in e["decision_fallback"]:
        run = run + 1 if held else 0
        best = max(best, run)
    return best


def _collisions(runs):
    """Collisions of the model-based filters: count, lift-from-slider tasks, and
    collisions in a rollout whose last decision was a fallback hold."""
    print("\nCollisions (1000 scenes)")
    for label in ("ShortStop", "MPC-Filter", "CBF"):
        hit = [e for e in runs[label].values() if e["violated"]]
        slider = sum(e["task_name"].startswith("lift") and e["task_name"].endswith("slider") for e in hit)
        held = sum(bool(e["decision_fallback"]) and e["decision_fallback"][-1] for e in hit)
        print(f"  {label:12s} {len(hit)} collisions, {slider} lifting a block from the slider, {held} during a fallback hold")


def _added_ms(label, e):
    """select + g(a) for the ranking shields, select for MPC/CBF (README Metrics)."""
    if label == "Unshielded":
        return []
    if label in ("MPC-Filter", "CBF"):
        return e["select_ms"]
    return [s + p for s, p in zip(e["select_ms"], e["progress_ms"])]


def _soundness():
    path = FINAL / "shortstop_hold_soundness_smoke.jsonl"
    if not path.exists():
        return
    rows = [json.loads(line) for line in open(path)]
    main_run = _load("shortstop_hold")
    same = sum(all(r[f] == main_run[(r["scene_id"], r["seed_k"])][f] for f in ("reached", "violated", "steps_taken"))
               for r in rows)
    s = np.array([x for r in rows for x in r["soundness"]])
    k, certified, true, dev = s[:, 1].astype(int), s[:, 2], s[:, 3], s[:, 4]
    eps = np.array([model_error_at(MODEL_ERROR, kk) for kk in k])
    print(f"\nSoundness: {len(rows)} scenes (outcome identical to the main run: {same}/{len(rows)}), {len(s)} executed "
          f"certified rows; arm outside the tube {np.mean(dev > eps):.4f}; true robustness below the certified one "
          f"{np.mean(true < certified - 1e-9):.4f}; certified rows that collided {int((true < 0).sum())}")
    for kk in sorted(set(k)):
        m = k == kk
        print(f"  k={kk}: n={m.sum()} deviation p99 {1e3 * np.percentile(dev[m], 99):.1f} mm, "
              f"max {1e3 * dev[m].max():.1f} mm, eps {1e3 * model_error_at(MODEL_ERROR, kk):.1f} mm, "
              f"outside {np.mean(dev[m] > eps[m]):.4f}")
    for r in rows:
        for row in r["soundness"]:
            if row[3] < 0:
                print(f"  collided certified row: {r['scene_id']} {r['task_name']} k={int(row[1])}, "
                      f"deviation {1e3 * row[4]:.1f} mm vs eps {1e3 * model_error_at(MODEL_ERROR, int(row[1])):.1f} mm")


def _tube_bound():
    """Corollary 2: with j_k the rank of the calibrated percentile among the n_k
    calibration errors (fixed by n_k before calibration) and eps_k at least that
    value, P[e_k > eps_k] <= delta_k = 1 - j_k / (n_k + 1); the tube holds at a
    decision with probability >= 1 - sum_k delta_k."""
    samples = np.load(CALIBRATION / "samples.npz")
    quantile = json.loads((CALIBRATION / "calibration.json").read_text())["quantile"]
    print(f"\nStatistical tube bound (calibration {CALIBRATION.name}, {quantile:.0%} percentile x safety factor)")
    total = 0.0
    for k in range(1, HORIZON_STEPS + 1):
        e = np.sort(samples[f"pose_frame_k{k}"])
        n, eps = len(e), model_error_at(MODEL_ERROR, k)
        j = int(np.floor(quantile * (n - 1))) + 1  # rank of np.percentile's lower order statistic
        assert e[j - 1] <= eps
        delta = 1 - j / (n + 1)
        total += delta
        print(f"  k={k}: n={n} j={j} delta {100 * delta:.3f} %, calibration errors above eps "
              f"{100 * np.mean(e > eps):.2f} %")
    print(f"  per-decision soundness probability >= {100 * (1 - total):.2f} % (H = {HORIZON_STEPS})")


def _coverage():
    scenes = json.loads(SCENES.read_text())["scenes"]
    print("\nTest scene coverage: " + ", ".join(
        f"first {n}: {len({s['task_name'] for s in scenes[:n]})} tasks, "
        f"{len({s['initial_state_index'] for s in scenes[:n]})} initial states" for n in (200, 300, len(scenes))))


if __name__ == "__main__":
    main()
