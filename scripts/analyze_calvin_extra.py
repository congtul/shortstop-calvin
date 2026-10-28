"""Ablation ladder, horizon sweep and delay stress test (docs/calvin/lab_notebook.md,
registered sections): safe success / violation with 95% bootstrap CIs,
activation, fallback, added latency and paired bootstrap tests, as LaTeX
table bodies for the paper (docs/paper/shortstop/tables/). Runs that are not
finished yet are printed as "pending".

    python scripts/analyze_calvin_extra.py
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from shortstop.calvin.baseline_runner import rollout_metrics  # noqa: E402

FINAL = REPO_ROOT / "outputs" / "calvin_final"
TABLES = REPO_ROOT / "docs" / "paper" / "shortstop" / "tables"
SCENES = REPO_ROOT / "shortstop" / "calvin" / "data" / "scenes" / "final_scenes.json"
N_BOOT = 10000

LADDER = [("L0", "MDT policy, no shield", "unshielded"),
          ("L1", "+ certify (tube, margin) + fallback, $K{=}1$", "reach_only_hold_abl_L1_smoke"),
          ("L2", "+ $K{=}4$ candidates (policy order)", "reach_only_hold_abl_L2_smoke"),
          ("L3", "+ progress score $g(a)$", "reach_only_hold_abl_L3_smoke"),
          ("L4", "+ counterexample repair (\\textsc{ShortStop})", "shortstop_hold")]
HORIZONS = [(4, "shortstop_hold"), (6, "shortstop_hold_H6_smoke"), (8, "shortstop_hold_H8_smoke")]
DELAY_METHODS = [("\\textsc{ShortStop}", "shortstop_hold"), ("MPC-Filter", "mpc_filter_hold"), ("CBF", "cbf_hold")]


def _scene_ids(n):
    return [s["scene_id"] for s in json.loads(SCENES.read_text())["scenes"][:n]]


def _load(name, ids):
    """Rows of run `name` for `ids` (in that order), or None if unfinished."""
    path = FINAL / f"{name}.jsonl"
    if not path.exists():
        return None
    rows = {r["scene_id"]: r for r in map(json.loads, path.read_text().splitlines()) if r["seed_k"] == 0}
    return [rows[i] for i in ids] if all(i in rows for i in ids) else None


def _safe(rows):
    return np.array([r["reached"] and not r["violated"] for r in rows], dtype=float)


def _viol(rows):
    return np.array([r["violated"] for r in rows], dtype=float)


def _ci(x, boot):
    m = x[boot].mean(1)
    return f"{100 * x.mean():.1f} [{100 * np.percentile(m, 2.5):.1f}, {100 * np.percentile(m, 97.5):.1f}]"


def _paired(a, b, boot):
    """(mean a - b in points, two-sided paired bootstrap p)."""
    d = (a - b)[boot].mean(1)
    return 100 * (a - b).mean(), min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))


def _shield_cols(rows):
    if "n_decisions" not in rows[0]:
        return "--", "--", "--"
    m = rollout_metrics(rows)
    return f"{100 * m['activation']:.1f}", f"{100 * m['fallback']:.1f}", f"{m['added_ms']['p95']:.0f}"


def _p(p):
    return "$<$.001" if p < 0.001 else f"{p:.3f}".lstrip("0")


def ablation(boot, ids):
    lines, prev = [], None
    for rung, label, name in LADDER:
        rows = _load(name, ids)
        if rows is None:
            lines.append(f"{rung} & {label} & \\multicolumn{{6}}{{c}}{{pending}} \\\\")
            prev = None
            continue
        act, fb, p95 = _shield_cols(rows)
        if prev is None:
            delta = "--"
        else:
            d, p = _paired(_safe(rows), _safe(prev), boot)
            delta = f"${d:+.1f}$ ({_p(p)})"
        lines.append(f"{rung} & {label} & {_ci(_safe(rows), boot)} & {100 * _viol(rows).mean():.1f} & "
                     f"{act} & {fb} & {p95} & {delta} \\\\")
        prev = rows
    return lines


def horizon(boot, ids):
    lines, base = [], _load(HORIZONS[0][1], ids)
    for h, name in HORIZONS:
        rows = _load(name, ids)
        if rows is None:
            lines.append(f"{h} & \\multicolumn{{6}}{{c}}{{pending}} \\\\")
            continue
        act, fb, p95 = _shield_cols(rows)
        if h == HORIZONS[0][0]:
            delta = "--"
        else:
            d, p = _paired(_safe(rows), _safe(base), boot)
            delta = f"${d:+.1f}$ ({_p(p)})"
        lines.append(f"{h} & {_ci(_safe(rows), boot)} & {100 * _viol(rows).mean():.1f} & {act} & {fb} & {p95} & "
                     f"{delta} \\\\")
    return lines


def delay(boot, ids):
    lines = []
    for d in (1, 2, 3):
        unsh = _load(f"unshielded_delay{d}_smoke", ids)
        u = "pending" if unsh is None else f"{100 * _safe(unsh).mean():.1f} / {100 * _viol(unsh).mean():.1f}"
        lines.append(f"\\multirow{{4}}{{*}}{{{d}}} & Unshielded & {u} & -- & -- & -- \\\\")
        for label, name in DELAY_METHODS:
            naive, aware = (_load(f"{name}_delay{d}_{mode}_smoke", ids) for mode in ("naive", "aware"))
            cells = ["pending" if rows is None else f"{100 * _safe(rows).mean():.1f} / {100 * _viol(rows).mean():.1f}"
                     for rows in (naive, aware)]
            if naive is None or aware is None:
                tests = ["--", "--"]
            else:
                tests = [f"${dd:+.1f}$ ({_p(pp)})" for dd, pp in
                         (_paired(_safe(aware), _safe(naive), boot), _paired(_viol(aware), _viol(naive), boot))]
            lines.append(f" & {label} & {cells[0]} & {cells[1]} & {tests[0]} & {tests[1]} \\\\")
        if d < 3:
            lines.append("\\midrule")
    return lines


def delay_vs_baselines(boot, ids):
    """ShortStop - baseline, aware mode, per delay (Markdown, for the text)."""
    out = []
    for d in (1, 2, 3):
        ss = _load(f"shortstop_hold_delay{d}_aware_smoke", ids)
        for label, name in DELAY_METHODS[1:]:
            other = _load(f"{name}_delay{d}_aware_smoke", ids)
            if ss is None or other is None:
                continue
            ds, ps = _paired(_safe(ss), _safe(other), boot)
            dv, pv = _paired(_viol(ss), _viol(other), boot)
            out.append(f"d={d} ShortStop - {label}: safe success {ds:+.1f} (p={ps:.3f}), violation {dv:+.1f} "
                       f"(p={pv:.3f})")
    return out


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    for name, fn, n in (("ablation", ablation, 300), ("horizon", horizon, 300), ("delay", delay, 200)):
        ids = _scene_ids(n)
        boot = np.random.default_rng(0).integers(0, n, (N_BOOT, n))
        lines = fn(boot, ids)
        (TABLES / f"{name}.tex").write_text("\n".join(lines) + "\n")
        print(f"== {name} (first {n} final scenes)\n" + "\n".join(lines))
    ids = _scene_ids(200)
    print("\n".join(delay_vs_baselines(np.random.default_rng(0).integers(0, 200, (N_BOOT, 200)), ids)))


if __name__ == "__main__":
    main()
