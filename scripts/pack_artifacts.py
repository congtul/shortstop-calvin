"""Build the data archives deposited with the paper (dist/*.zip), each with a
README and a SHA-256 manifest of its files.

    python scripts/pack_artifacts.py

- shortstop_calvin_results.zip: every rollout behind the paper's tables and
  figures (final, soundness, delay, ablation, horizon), the validation runs
  that fixed H, K and the tuned parameters (with the K probe in the final
  configuration), the candidate-spread log, the model-error and joint-hold
  calibration runs and the stage logs. Unzip in the repo root;
  scripts/analyze_calvin_*.py read it.
- shortstop_calvin_reference_rollouts.zip: the obstacle-free unshielded
  rollouts the scene builder samples obstacles from (needed only to rebuild
  the frozen scenes).
- shortstop_calvin_superseded_v1.zip: the first final run and its tuning,
  superseded by v2 (docs/calvin/theory_audit.md); kept for the record.
"""
import hashlib
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "outputs"
DIST = REPO_ROOT / "dist"

FINAL_METHODS = ["unshielded", "shortstop_hold", "mpc_filter_hold", "cbf_hold", "conf_thresh_hold", "stl_monitor_hold"]
RUN_PREFIXES = (
    FINAL_METHODS
    + ["shortstop_hold_soundness_smoke"]
    + [f"unshielded_delay{d}_smoke" for d in (1, 2, 3)]
    + [f"{m}_hold_delay{d}_{mode}_smoke" for d in (1, 2, 3) for m in ("shortstop", "mpc_filter", "cbf")
       for mode in ("naive", "aware")]
    + [f"reach_only_hold_abl_L{i}_smoke" for i in (1, 2, 3)]
    + [f"shortstop_hold_H{h}_smoke" for h in (6, 8)]
)
VAL_PREFIXES = ([f"{m}_hold_H{h}" for m in ("shortstop", "mpc_filter", "cbf") for h in (4, 6, 8)]
                + [f"{m}_hold_gate_smoke" for m in ("shortstop", "mpc_filter", "cbf")] + ["unshielded"]
                + ["shortstop_freeze_c5_K1", "shortstop_freeze_c5_K4", "shortstop_freeze_c5"]  # early K probe
                + [f"shortstop_hold_kprobe_K{k}" for k in (1, 4, 8)]  # K probe, final configuration
                + ["conf_thresh_hold_spread"])  # candidate spread (Conf-Thresh disagreement log)
STAGE_LOGS = {  # stage log in outputs/pipeline_v2 -> name in the archive
    "pipeline.out": "pre_final.out", "final.out": "final.out", "e2_before_reboot.out": "delay_part1.out",
    "e2.out": "delay.out", "ablation_smoke.out": "ablation_smoke.out",
    "ablation_full_part1.out": "ablation_full_part1.out", "ablation_full.out": "ablation_full.out",
    "chosen_H.txt": "chosen_H.txt",
}

README_RESULTS = """ShortStop on CALVIN: rollouts behind every table and figure of the paper.

Unzip in the root of the code repository (it creates outputs/...), then
    python scripts/analyze_calvin_final.py    # Table 1, paired tests, soundness, tube bound
    python scripts/analyze_calvin_extra.py    # Tables 2-3, horizon sweep
    python scripts/analyze_calvin_k_probe.py  # K probe (40 val scenes, Sect. 5)
One JSON line per rollout; fields are described in docs/calvin/README.md
(protocol) and shortstop/calvin/experiment.py (run_scene). *_summary.json are
the runner's pooled metrics, *.log the per-run logs (a run restarted after an
interruption has several). outputs/pipeline/ holds the stage logs; the delay
test was interrupted by an OS restart and resumed (delay_part1.out, delay.out),
the ablation was paused once (ablation_full_part1.out); k_probe.out is the
K probe's stage log. The published
unshielded rows come from the first final run (identical code path; its
unused fallback label reads "freeze").
"""
README_REFERENCE = """ShortStop on CALVIN: obstacle-free unshielded reference rollouts
(scripts/collect_calvin_unshielded_rollouts.py). The scene builder samples
obstacle positions and checks hit rates against them. Needed only to rebuild
the frozen scenes; unzip in the repository root (creates outputs/unshielded_rollouts/).
"""
README_V1 = """ShortStop on CALVIN: the first final run (v1: one constant model error,
freeze fallback) and its tuning. Superseded by v2 after the theory audit
(docs/calvin/theory_audit.md); the paper reports v2 only. Kept for the record.
"""

LICENSE = "\nLicense: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).\n"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _run_files(directory, prefixes):
    """jsonl, summary and logs of exactly these runs (not of runs whose name
    merely starts with a prefix)."""
    files = []
    for prefix in prefixes:
        for path in sorted(directory.glob(f"{prefix}*")):
            rest = path.name[len(prefix):]
            if rest in (".jsonl", "_summary.json") or (rest.startswith("_2026") and rest.endswith(".log")):
                files.append(path)
        if not (directory / f"{prefix}.jsonl").exists():
            raise FileNotFoundError(directory / f"{prefix}.jsonl")
    return files


TEXT = {".log", ".out", ".sh", ".json", ".txt", ".jsonl"}


def _content(src):
    """File bytes; text files with the machine's absolute paths replaced."""
    data = src.read_bytes()
    if src.suffix in TEXT:
        data = data.replace(str(REPO_ROOT).encode(), b"<repo>").replace(str(Path.home()).encode(), b"<home>")
    return data


def _write(name, readme, members, docs_dir="outputs"):
    """members: list of (source path, name in archive); the README and the
    manifest go to <docs_dir>/<archive name>.{README.txt,sha256}."""
    DIST.mkdir(exist_ok=True)
    manifest = []
    with zipfile.ZipFile(DIST / name, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for src, arc in members:
            data = _content(src)
            manifest.append(f"{hashlib.sha256(data).hexdigest()}  {arc}")
            z.writestr(zipfile.ZipInfo(arc, date_time=(2026, 10, 17, 0, 0, 0)), data, zipfile.ZIP_DEFLATED)
        stem = f"{docs_dir}/{name[:-len('.zip')]}"
        z.writestr(zipfile.ZipInfo(f"{stem}.README.txt", date_time=(2026, 10, 17, 0, 0, 0)), readme + LICENSE)
        z.writestr(zipfile.ZipInfo(f"{stem}.sha256", date_time=(2026, 10, 17, 0, 0, 0)), "\n".join(manifest) + "\n")
    size = (DIST / name).stat().st_size / 1e6
    print(f"{name}: {len(members)} files, {size:.1f} MB, sha256 {_sha256(DIST / name)}")


def main():
    rel = lambda p: str(p.relative_to(REPO_ROOT))  # noqa: E731
    results = [(p, rel(p)) for p in _run_files(OUT / "calvin_final", RUN_PREFIXES)]
    results += [(p, rel(p)) for p in _run_files(OUT / "calvin_probe_local_cluster", VAL_PREFIXES)]
    results += [(p, rel(p)) for p in _run_files(OUT / "calvin_tuning_probe_local_cluster",
                                                 ["cbf_hold_v2", "conf_thresh_hold_v2"])]
    results += [(OUT / "calvin_tuning_probe_local_cluster" / f, f"outputs/calvin_tuning_probe_local_cluster/{f}")
                for f in ("cbf_hold_v2_chosen.json", "conf_thresh_hold_v2_chosen.json")]
    for run in ("run_20261012_120501", "run_20261012_164600"):  # epsilon_k; joint-hold drift over 300 steps
        results += [(p, rel(p)) for p in sorted((OUT / "calibrate_arm_model_error_runs" / run).iterdir())]
    results += [(OUT / "pipeline_v2" / src, f"outputs/pipeline/{dst}") for src, dst in STAGE_LOGS.items()]
    results.append((OUT / "calvin_probe_local_cluster" / "kprobe.out", "outputs/pipeline/k_probe.out"))
    _write("shortstop_calvin_results.zip", README_RESULTS, results)

    reference = [(p, rel(p)) for p in sorted((OUT / "unshielded_rollouts").rglob("*")) if p.is_file()]
    _write("shortstop_calvin_reference_rollouts.zip", README_REFERENCE, reference)

    v1 = [(p, rel(p)) for p in sorted((OUT / "calvin_final").glob("*_freeze*"))]
    v1 += [(p, rel(p)) for p in sorted((OUT / "calvin_tuning_probe_local_cluster").glob("*_freeze*"))]
    v1 += [(OUT / "calvin_final" / "run_final.sh", "outputs/calvin_final/run_final.sh"),
           (OUT / "calvin_final" / "run_final.out", "outputs/calvin_final/run_final.out")]
    _write("shortstop_calvin_superseded_v1.zip", README_V1, v1)


if __name__ == "__main__":
    main()
