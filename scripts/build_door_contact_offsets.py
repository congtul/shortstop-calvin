"""Write shortstop/calvin/data/door_contact_offsets.json: for each
move_door_rel task, where the fingertip works the door (offset from the
moving link's box_min), measured on the task's successful reference
rollouts (scene_manifest.door_contact_offset). No simulator.

    cd SHORTSTOP
    python scripts/build_door_contact_offsets.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "mdt_policy"))

from shortstop.calvin import scene_manifest as sm  # noqa: E402
from shortstop.calvin.independent_tasks import _enumerate_initial_states  # noqa: E402
from shortstop.calvin.static_targets import DOOR_CONTACT_PATH  # noqa: E402

TASKS_YAML = REPO_ROOT / "mdt_policy" / "calvin_env" / "conf" / "tasks" / "new_playtable_tasks.yaml"
ROLLOUTS_DIR = REPO_ROOT / "outputs" / "unshielded_rollouts"


def main():
    from mdt.evaluation.multistep_sequences import tasks, valid_task
    from mdt.evaluation.utils import get_env_state_for_initial_condition

    specs = yaml.safe_load(TASKS_YAML.read_text())["tasks"]
    states = _enumerate_initial_states()
    out = {}
    for name, spec in specs.items():
        if spec[0] != "move_door_rel" or not any(valid_task(s, tasks[name]) for s in states):
            continue
        joint, threshold = spec[1], spec[2]
        starts = {float(get_env_state_for_initial_condition(s)[1][sm._SCENE_OBS_JOINT[joint]])
                  for s in states if valid_task(s, tasks[name])}
        q_end = starts.pop() + threshold
        offset, std, n = sm.door_contact_offset(np.load(ROLLOUTS_DIR / f"rollouts_{name}.npz"), joint, q_end)
        out[name] = {"joint": joint, "offset_from_box_min": offset.tolist(), "std": std.tolist(),
                     "n_successful": n, "q_end": q_end}
        print(f"[door] {name:18s} fingertip offset {np.round(offset, 4).tolist()} std {np.round(std, 4).tolist()} n={n}")
    DOOR_CONTACT_PATH.write_text(json.dumps(out, indent=1))
    print(f"[door] wrote {DOOR_CONTACT_PATH}")


if __name__ == "__main__":
    main()
