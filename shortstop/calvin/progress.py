"""Task-progress score g(a) for CALVIN.

Select picks the admissible candidate with the highest g(a). The paper gives
no formula for a language-conditioned domain, so this is the project's own:

    g(a) = -distance(predicted TCP at the end of chunk a, current goal)

- TCP = the fingertip (robot_geometry.gripper_tip_position), the point
  CALVIN's actions move, predicted with the shields' kinematic model
  (reach.propagate_arm_tube).
- The goal comes from shortstop.calvin.task_goals, the same definition the
  scene builder's feasibility planner uses: the contact point on the object
  until the object starts to move, then that point in the finished state.

Limits: no orientation (rotate_object keeps its contact goal); the score is
proximity, not task completion. Tasks without a goal score 0.0 for every
candidate (neutral).
"""
import numpy as np

from .reach import propagate_arm_tube
from .robot_geometry import gripper_tip_position, point_to_box_distance
from .task_goals import SubtaskGoals


def subtask_goals(task_oracle, subtask, start_info, start_joint_angles,
                  base_position=(0.0, 0.0, 0.0), base_orientation=(0.0, 0.0, 0.0, 1.0)):
    """SubtaskGoals for `subtask` from its start state, or None when the task
    has no goal (no config args or an unsupported kind)."""
    task_fn = task_oracle.tasks.get(subtask)
    if task_fn is None or not getattr(task_fn, "args", None):
        return None
    spec = (getattr(task_fn.func, "__name__", ""), *task_fn.args)
    try:
        return SubtaskGoals(subtask, spec, start_info["scene_info"], gripper_tip_position(start_joint_angles),
                            base_position, base_orientation)
    except ValueError:
        return None


def calvin_progress_scores(joint_angles, candidates, goal_box, tcp_euler=None):
    """g(a) per candidate: minus the distance from its predicted TCP after the
    whole chunk (f-hat; pose model with `tcp_euler`) to `goal_box` =
    (box_min, box_max) in the local frame. All 0.0 when `goal_box` is None."""
    if goal_box is None:
        return [0.0] * len(candidates)
    b_min, b_max = (np.asarray(b, dtype=float) for b in goal_box)
    scores = []
    for chunk in candidates:
        tube = propagate_arm_tube(joint_angles, chunk, w_bar=0.0, model_error=0.0, tcp_euler=tcp_euler)
        scores.append(-point_to_box_distance(tube[-1]["fingertip"].b, b_min, b_max))
    return scores
