"""Closed loop on one probe scene (freeze, seed 0) for ShortStop repair modes."""
import json, sys, time
from pathlib import Path
import calvin_env  # noqa: F401
import hydra
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "mdt_policy"))
from shortstop.calvin import scene_manifest as sm  # noqa: E402
from shortstop.calvin.baseline_runner import make_shield, run_scene, setup_env_and_policy  # noqa: E402
from shortstop.calvin.config import N_CANDIDATES  # noqa: E402

@hydra.main(config_path=str(REPO / "mdt_policy/conf"), config_name="mdt_evaluate")
def main(cfg):
    from mdt.evaluation.utils import get_env_state_for_initial_condition
    split = cfg.scene_id.rsplit("_", 1)[0]
    _, scenes = sm.load_scenes(REPO / f"shortstop/calvin/data/scenes/{split}_scenes.json", get_env_state_for_initial_condition)
    scene = next(s for s in scenes if s["scene_id"] == cfg.scene_id)
    env, policy, task_oracle, ann, lang = setup_env_and_policy(cfg, N_CANDIDATES)
    base = json.load(open(REPO / "outputs/calvin_tuning/shortstop_retreat0.02_chosen.json"))["params"]
    for mode in str(cfg.get("modes", "polyak,ce_qp")).split(","):
        shield = make_shield("shortstop", {**base, "repair_mode": mode})
        t0 = time.time()
        row = run_scene(env, policy, task_oracle, ann, lang, get_env_state_for_initial_condition, scene, 0, shield, cfg.ep_len)
        print(f"[one_scene] {cfg.scene_id} repair={mode} reached={row['reached']} violated={row['violated']} "
              f"steps={row['steps_taken']} fallbacks={row['n_fallback']}/{row['n_decisions']} "
              f"min_clear={row['min_clearance']:.3f} {time.time() - t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
