# CALVIN task set — 27 task types

This project evaluates **one task per sequence** (independent single-task
trials, `shortstop/calvin/independent_tasks.py`), not CALVIN's official
5-task chains. A task can only appear if it can start from one of CALVIN's
own fresh initial states, which leaves **27 of CALVIN's 34 task types**.
`stack_block`/`unstack_block` are additionally dropped on purpose
(`shortstop.calvin.rollout_collection.EXCLUDED_TASKS`).

## The 27 tasks

| Group | Tasks | # |
|---|---|---|
| Drawer / slider | `open_drawer`, `close_drawer`, `move_slider_left`, `move_slider_right` | 4 |
| Lights | `turn_on_led`, `turn_off_led`, `turn_on_lightbulb`, `turn_off_lightbulb` | 4 |
| Push block | `push_{red,blue,pink}_block_left`, `push_{red,blue,pink}_block_right`, `push_into_drawer` | 7 |
| Rotate block | `rotate_{red,blue,pink}_block_left`, `rotate_{red,blue,pink}_block_right` | 6 |
| Lift block | `lift_{red,blue,pink}_block_table`, `lift_{red,blue,pink}_block_slider` | 6 |

## The 7 not tested, and why

CALVIN's initial-state generator (`get_sequences`, copied in
`independent_tasks._enumerate_initial_states`) only places blocks on the
`table`/`slider_left`/`slider_right` and always starts with `grasped: 0`.

| Task | Needs | Why it can't start fresh |
|---|---|---|
| `lift_{red,blue,pink}_block_drawer` | block in the drawer | blocks never start in the drawer |
| `place_in_slider`, `place_in_drawer` | a block already grasped | `grasped` is always 0 at start |
| `stack_block` | a block already grasped | same; also excluded on purpose |
| `unstack_block` | two blocks already stacked | never a start state; also excluded on purpose |

In CALVIN's official chains these only appear after an earlier subtask set
them up (e.g. `lift_*` → `place_in_*`, `push_into_drawer` →
`lift_*_block_drawer`). Testing them here would need custom start states
(block placed inside the drawer) or 2-task chains scored on the 2nd task —
not done; decided 2026-10-04 to keep the 27.

Recompute the split any time with
`rollout_collection.eligible_task_names(tasks, _enumerate_initial_states(), valid_task)`.
