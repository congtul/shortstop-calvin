"""Where the fingertip (TCP) has to go for each CALVIN task -- one
definition shared by the scene builder's feasibility check
(shortstop.calvin.feasibility_planner via scene_manifest) and by g(a)
(shortstop.calvin.progress), so the score steers toward exactly what the
planner proved reachable.

Each task has two goals, following how the robot actually works the
object:
- CONTACT (phase 1, reaching): where the fingertip touches the object --
  the block's center, the drawer/slider contact point (handle, or front
  face when closing; measured, static_targets.door_contact_offset), the
  switch lever, the button.
- COMPLETION (phase 2, working the object): the same contact point once the
  object is in its finished state -- block pushed 10 cm / lifted 7 cm,
  drawer moved by its threshold, switch at the other end, button pressed;
  rotate_object's completion is a wrist turn with no position target.

`SubtaskGoals` holds both for one subtask (fixed from the subtask's start
state) and switches to phase 2 once the object starts to move -- read from
the live scene_info, so no guessing about contact.

Everything returned is in the robot-base-LOCAL frame (the predictor's).
"""
import numpy as np

from .feasibility_planner import TaskGoal
from .robot_geometry import box_to_local_frame, to_local_frame
from .static_targets import door_contact_offset, static_target_box_at, static_target_joint_range

# Extra lift height (m) on top of lift_object's own threshold for the
# completion goal, so "completed" isn't right on the success boundary.
LIFT_EXTRA = 0.02

# Phase 2 starts once the object has visibly moved from its start state:
# a block by more than this (m), a joint by more than this fraction of its
# travel (button 2 mm, switch 8.8 mm, drawer 2.2 cm, slider 2.8 cm).
BLOCK_MOVED = 0.01
JOINT_MOVED_FRACTION = 0.1

# Light -> its actuator joint (the part the hand works), calvin_scene_D.
LIGHT_ACTUATOR = {"lightbulb": "base__switch", "led": "base__button"}

# Joint -> where its live value sits in scene_info (section, entry, field).
_JOINT_STATE = {
    "base__slide": ("doors", "base__slide", "current_state"),
    "base__drawer": ("doors", "base__drawer", "current_state"),
    "base__switch": ("switches", "base__switch", "joint_state"),
    "base__button": ("buttons", "base__button", "joint_state"),
}
# Cache key the static target boxes of each joint were stored under.
_JOINT_CACHE_KEY = {"base__slide": "base__slide", "base__drawer": "base__drawer",
                    "base__switch": "lightbulb", "base__button": "led"}


def block_position(scene_info, name):
    return np.asarray(scene_info["movable_objects"][name]["current_pos"], dtype=float)


def joint_value(scene_info, joint):
    section, entry, field = _JOINT_STATE[joint]
    return float(scene_info[section][entry][field])


def block_on_table(scene_info, name):
    # Table spots have y = -0.12; slider spots y = +0.092 (same height).
    return block_position(scene_info, name)[1] < 0.0


def _required_box(cache_key, q):
    box = static_target_box_at(cache_key, q)
    if box is None:
        raise ValueError(f"no cached endpoint boxes for {cache_key!r} -- run scripts/precompute_static_target_sweeps.py")
    return box


def _required_range(cache_key):
    joint_range = static_target_joint_range(cache_key)
    if joint_range is None:
        raise ValueError(f"no cached joint range for {cache_key!r} -- run scripts/precompute_static_target_sweeps.py")
    return joint_range


def task_goals(task_name, spec, scene_info, start_tip_local, base_position, base_orientation,
               door_contact_offsets=None):
    """(contact TaskGoal, completion TaskGoal) in the local frame.

    `spec` = the task's config entry (calvin_env/conf/tasks/
    new_playtable_tasks.yaml): (function name, *args), e.g.
    ('move_door_rel', 'base__drawer', 0.12). `scene_info` = the state the
    goals are taken from (the subtask's start). `start_tip_local` = the
    fingertip at that moment (picks push_into_drawer's block).
    `door_contact_offsets` ({task_name: (3,)}) overrides the measured door
    contact points (default: static_targets.door_contact_offset). Each
    goal's `meta` names the object/joint that moves, for phase detection.
    Raises ValueError for unsupported tasks or missing static caches."""
    kind, *args = spec

    def point(world, **meta):
        p = to_local_frame(np.asarray(world, dtype=float), base_position, base_orientation)
        return TaskGoal(p, p.copy(), meta=meta)

    def box(world_box, **meta):
        b_min, b_max = box_to_local_frame(world_box[0], world_box[1], base_position, base_orientation)
        return TaskGoal(b_min, b_max, meta=meta)

    if kind == "rotate_object":
        obj, degrees = args
        contact = point(block_position(scene_info, obj), obj=obj)
        # Joint 7's axis is the flange z, pointing DOWN while the gripper
        # points down, so a world-CCW (+degrees) block turn is -degrees here.
        return contact, TaskGoal(wrist_turn=-np.radians(degrees), meta={"obj": obj, "degrees": degrees})
    if kind == "push_object":
        obj, dx, dy = args
        start = block_position(scene_info, obj)
        return point(start, obj=obj), point(start + np.array([dx, dy, 0.0]), obj=obj, pushed_by=[dx, dy])
    if kind == "lift_object":
        obj, dz = args[0], args[1]
        start = block_position(scene_info, obj)
        lift = dz + LIFT_EXTRA
        return point(start, obj=obj), point(start + np.array([0.0, 0.0, lift]), obj=obj, lifted_by=lift)
    if kind == "move_door_rel":
        joint, threshold = args
        q = joint_value(scene_info, joint)
        q_end = float(np.clip(q + threshold, *_required_range(joint)))
        offsets = door_contact_offsets if door_contact_offsets is not None else {}
        offset = offsets.get(task_name) if task_name in offsets else door_contact_offset(task_name)
        if offset is not None:
            offset = np.asarray(offset, dtype=float)
            return (point(_required_box(joint, q)[0] + offset, joint=joint, q=q, contact="measured"),
                    point(_required_box(joint, q_end)[0] + offset, joint=joint, q=q_end, contact="measured"))
        return (box(_required_box(joint, q), joint=joint, q=q),
                box(_required_box(joint, q_end), joint=joint, q=q_end))
    if kind == "toggle_light":
        light = args[0]
        actuator = LIGHT_ACTUATOR[light]
        q = joint_value(scene_info, actuator)
        ll, ul = _required_range(light)
        # Switch: flip to the far end of its travel. Button: press (to ul).
        q_end = (ll if q > 0.5 * (ll + ul) else ul) if actuator == "base__switch" else ul
        return (box(_required_box(light, q), light=light, joint=actuator, q=q),
                box(_required_box(light, q_end), light=light, joint=actuator, q=q_end))
    if kind == "push_object_into":
        objects = args[0]
        on_table = [o for o in objects if block_on_table(scene_info, o)]
        if not on_table:
            raise ValueError(f"{task_name}: none of {objects} is on the table in this scene")
        tip = np.asarray(start_tip_local, dtype=float)
        obj = min(on_table, key=lambda o: np.linalg.norm(
            to_local_frame(block_position(scene_info, o), base_position, base_orientation) - tip))
        drawer_q = joint_value(scene_info, "base__drawer")
        return (point(block_position(scene_info, obj), obj=obj),
                box(_required_box("base__drawer", drawer_q), joint="base__drawer", q=drawer_q))
    raise ValueError(f"unsupported task {task_name!r} ({kind})")


def object_moved(contact_meta, start_scene_info, scene_info):
    """Has the task's object started to move since the subtask began?"""
    if "obj" in contact_meta:
        obj = contact_meta["obj"]
        return float(np.linalg.norm(block_position(scene_info, obj) - block_position(start_scene_info, obj))) > BLOCK_MOVED
    joint = contact_meta["joint"]
    ll, ul = _required_range(_JOINT_CACHE_KEY[joint])
    return abs(joint_value(scene_info, joint) - joint_value(start_scene_info, joint)) > JOINT_MOVED_FRACTION * (ul - ll)


class SubtaskGoals:
    """g(a)'s target for one subtask: the contact goal until the object
    starts to move, then the completion goal (latched -- no switching
    back). rotate_object keeps the contact goal (its completion is a wrist
    turn, which a position score can't express)."""

    def __init__(self, task_name, spec, start_scene_info, start_tip_local, base_position, base_orientation):
        self.contact, self.completion = task_goals(
            task_name, spec, start_scene_info, start_tip_local, base_position, base_orientation)
        self.start_scene_info = start_scene_info
        self.phase = 1

    def target(self, scene_info):
        """(box_min, box_max) in the local frame for the current phase."""
        if self.phase == 1 and object_moved(self.contact.meta, self.start_scene_info, scene_info):
            self.phase = 2
        goal = self.completion if self.phase == 2 and self.completion.box_min is not None else self.contact
        return goal.box_min, goal.box_max
