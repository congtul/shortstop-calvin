"""Re-run ShortStop (pivot params, freeze) on one probe scene and save every
decision where repair failed (total fallback): joint angles, the 8 candidates,
scores and the obstacles -> capture_<scene>.npz."""
import json
import sys
from pathlib import Path

import calvin_env  # noqa: F401
import hydra
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "mdt_policy"))

from shortstop.calvin import scene_manifest as sm  # noqa: E402
from shortstop.calvin.baseline_runner import make_shield, run_scene, setup_env_and_policy  # noqa: E402
from shortstop.calvin.config import N_CANDIDATES  # noqa: E402


@hydra.main(config_path=str(REPO / "mdt_policy/conf"), config_name="mdt_evaluate")
def main(cfg):
    scene_id = cfg.scene_id
    split = scene_id.rsplit("_", 1)[0]
    from mdt.evaluation.utils import get_env_state_for_initial_condition
    _, scenes = sm.load_scenes(REPO / f"shortstop/calvin/data/scenes/{split}_scenes.json", get_env_state_for_initial_condition)
    scene = next(s for s in scenes if s["scene_id"] == scene_id)
    env, policy, task_oracle, ann, lang = setup_env_and_policy(cfg, N_CANDIDATES)
    params = json.load(open(REPO / "outputs/calvin_tuning/shortstop_retreat0.02_chosen.json"))["params"]
    shield = make_shield("shortstop", params)
    captured = []
    original = shield.select

    def select(q, candidates, scores):
        chunk, info = original(q, candidates, scores)
        if info["fallback"] and len(captured) < int(cfg.get("max_capture", 40)):
            captured.append({"q": np.asarray(q), "candidates": np.asarray(candidates), "scores": np.asarray(scores)})
        return chunk, info

    shield.select = select
    row = run_scene(env, policy, task_oracle, ann, lang, get_env_state_for_initial_condition, scene, 0, shield,
                    cfg.ep_len, fallback="freeze")
    print("[capture]", scene_id, "reached", row["reached"], "violated", row["violated"], "steps", row["steps_taken"],
          "fallbacks", row["n_fallback"], "captured", len(captured), flush=True)
    out = REPO / "outputs/mechanism_study" / f"capture_{scene_id}.npz"
    np.savez(out, q=np.array([c["q"] for c in captured]), candidates=np.array([c["candidates"] for c in captured]),
             scores=np.array([c["scores"] for c in captured]),
             obstacle_centers=np.array([o.center for o in scene["obstacles"]]),
             obstacle_radii=np.array([o.radius for o in scene["obstacles"]]))
    print("[capture] wrote", out, flush=True)


if __name__ == "__main__":
    main()
