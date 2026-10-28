"""Compare rerun rollouts with the published ones, field by field.

    python scripts/verify_reproduction.py shortstop_hold cbf_hold ...

For each run name, reads outputs/calvin_final/<name>_repro_smoke.jsonl (made by
scripts/pipeline/repro_check.sh) and the published <name>.jsonl, and reports
every scene whose outcome fields differ. Timing fields are ignored; fields the
published rows do not have (added to the log later) are skipped. The published
unshielded rows come from the first final run, whose (unused) fallback label
was "freeze"; the unshielded policy never falls back, so that label is skipped.
"""
import json
import sys
from pathlib import Path

FINAL = Path(__file__).resolve().parents[1] / "outputs" / "calvin_final"
IGNORED = {"seconds", "select_ms", "added_ms", "progress_ms", "propose_ms"}


def _rows(path):
    return {r["scene_id"]: r for r in map(json.loads, path.read_text().splitlines()) if r["seed_k"] == 0}


def main(names):
    failed = False
    for name in names:
        published, rerun = _rows(FINAL / f"{name}.jsonl"), _rows(FINAL / f"{name}_repro_smoke.jsonl")
        for scene_id, row in sorted(rerun.items()):
            ref = published[scene_id]
            ignored = IGNORED | ({"fallback"} if ref.get("baseline") == "unshielded" else set())
            diff = [k for k in ref if k not in ignored and k in row and ref[k] != row[k]]
            failed |= bool(diff)
            print(f"{name:20s} {scene_id}  {'identical' if not diff else 'DIFFERS in ' + ', '.join(diff)}")
    print("REPRODUCED" if not failed else "NOT REPRODUCED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
