"""Print what CALVIN's `env.get_info()["scene_info"]` exposes at runtime
(block poses, door/switch/button joint values) and each task's config args
-- the fields shortstop.calvin.task_goals reads.

Run from WSL2, inside `mdt_env`:

    cd SHORTSTOP
    python scripts/verify_calvin_scene_info_fields.py
"""
import sys
from pathlib import Path

import calvin_env  # noqa: F401 -- import side effects only, matches every other CALVIN script
import hydra
from pytorch_lightning import seed_everything

REPO_ROOT = Path(__file__).resolve().parents[1]
MDT_POLICY_ROOT = REPO_ROOT / "mdt_policy"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(MDT_POLICY_ROOT))

from shortstop.calvin.baseline_runner import setup_env_and_policy  # noqa: E402
from shortstop.calvin.config import N_CANDIDATES  # noqa: E402

# Real CALVIN task names known (see precompute_static_target_sweeps.py's
# own DOOR_JOINT_NAMES/LIGHT_NAMES and tests/test_calvin_progress.py's own
# _FakeTaskOracle fixtures, e.g. "open_drawer"/"turn_on_led") to involve a
# move_door_rel/toggle_light subtask (Open item 2) or a rotate_object
# subtask on a block (Open item 1) -- any sequence containing at least one
# of these is a valid scene to inspect for both questions at once.
DOOR_TASK_NAMES = {"open_drawer", "close_drawer", "move_slider_left", "move_slider_right"}
LIGHT_TASK_NAMES = {"turn_on_lightbulb", "turn_off_lightbulb", "turn_on_led", "turn_off_led"}
ROTATE_TASK_NAMES = {"rotate_red_block_right", "rotate_red_block_left",
                      "rotate_blue_block_right", "rotate_blue_block_left",
                      "rotate_pink_block_right", "rotate_pink_block_left"}
TARGET_TASK_NAMES = DOOR_TASK_NAMES | LIGHT_TASK_NAMES | ROTATE_TASK_NAMES

# How many sequences to sample from get_sequences() while looking for one
# that contains a door/light/rotate task -- get_sequences's own task-type
# mix is random per call, so this may need raising if none of the first N
# happen to include one (unlikely at 200, but not impossible). If the
# first match found only covers one of the 2 open items (e.g. a door task
# but no rotate task), re-run -- this script inspects whatever movable_
# objects/doors happen to be present in scene_info regardless of which
# TARGET_TASK_NAMES entry triggered the match, so a lucky sequence often
# answers both at once anyway.
N_SEQUENCES_TO_SAMPLE = 200


@hydra.main(config_path="../mdt_policy/conf", config_name="mdt_evaluate")
def main(cfg):
    seed_everything(0, workers=True)
    env, policy, task_oracle, val_annotations, lang_embeddings = setup_env_and_policy(cfg, N_CANDIDATES)
    del policy, val_annotations, lang_embeddings  # task_oracle is also needed now, see below

    from mdt.evaluation.multistep_sequences import get_sequences
    from mdt.evaluation.utils import get_env_state_for_initial_condition

    found = None
    for initial_state, eval_sequence in get_sequences(N_SEQUENCES_TO_SAMPLE):
        if any(name in TARGET_TASK_NAMES for name in eval_sequence):
            found = (initial_state, eval_sequence)
            break
    if found is None:
        print(
            f"[verify] none of the {N_SEQUENCES_TO_SAMPLE} sampled sequences contain a "
            "door/light/rotate task -- re-run, or raise N_SEQUENCES_TO_SAMPLE above"
        )
        return

    initial_state, eval_sequence = found
    print(f"[verify] using sequence {eval_sequence}")
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    info = env.get_info()
    scene_info = info["scene_info"]
    print(f"[verify] scene_info top-level keys: {sorted(scene_info.keys())}")

    # --- Open item 2: live door/switch joint state? ---
    door_like_keys = [
        k for k in scene_info
        if any(word in k.lower() for word in ("door", "light", "switch", "button", "slider", "drawer"))
    ]
    if not door_like_keys:
        print(
            "[verify][open-item-2] no key in scene_info looks door/light/switch-related -- live "
            "joint state is likely NOT exposed this way"
        )
    for key in door_like_keys:
        print(f"[verify][open-item-2] scene_info[{key!r}] = {scene_info[key]!r}")

    # --- Open item 1: block orientation? ---
    movable_objects = scene_info.get("movable_objects", {})
    if not movable_objects:
        print("[verify][open-item-1] scene_info['movable_objects'] is empty for this scene -- re-run")
    for name, entry in movable_objects.items():
        print(f"[verify][open-item-1] movable_objects[{name!r}] keys={sorted(entry.keys())} -> {entry!r}")
    orientation_like_keys = {
        key for entry in movable_objects.values() for key in entry
        if any(word in key.lower() for word in ("orn", "orient", "quat", "rot", "euler"))
    }
    if not orientation_like_keys:
        print(
            "[verify][open-item-1] no orientation-like key found in any movable_objects entry -- "
            "current_pos is likely the ONLY field available"
        )
    else:
        print(f"[verify][open-item-1] orientation-like keys found: {sorted(orientation_like_keys)}")

    # --- Open item 1 follow-up: does rotate_object's own task config carry
    # a target angle/direction, like move_door_rel's (joint_name,
    # threshold)? task_oracle.tasks is the SAME {name: functools.partial(
    # base_fn, *args)} mapping shortstop.calvin.progress._target_object_
    # boxes already reads -- print every name this scene's eval_sequence
    # actually uses, plus every ROTATE_TASK_NAMES entry task_oracle knows
    # about (not just the one that happened to trigger this sequence
    # match), so one run answers the question for all 6 rotate variants
    # at once. ---
    print("[verify][open-item-1-followup] task_fn.args per task name (checking for a target angle/direction):")
    for name in sorted(set(eval_sequence) | ROTATE_TASK_NAMES):
        task_fn = task_oracle.tasks.get(name)
        if task_fn is None:
            print(f"    {name!r}: not in task_oracle.tasks")
            continue
        print(f"    {name!r}: args={task_fn.args!r}")

    print("[verify] full scene_info (for manual inspection, in case a field is nested/named unexpectedly):")
    print(scene_info)


if __name__ == "__main__":
    main()
