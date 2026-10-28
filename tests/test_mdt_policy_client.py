import numpy as np
import pytest
import torch

from shortstop.calvin.mdt_policy_client import ForwardOnlyPolicy, MockMDTPolicyClient


def test_mock_client_returns_k_chunks_of_the_right_shape():
    client = MockMDTPolicyClient(horizon=10, action_dim=7, n_candidates=8, rng=np.random.default_rng(0))
    candidates = client.propose(observation={})
    assert len(candidates) == 8
    for chunk in candidates:
        assert chunk.shape == (10, 7)


class _FakeMDTModel:
    """Minimal stand-in for MDTVAgent exposing exactly the attributes/
    methods ForwardOnlyPolicy.propose() calls (mirroring the real
    mdtv_agent.py:688 forward()'s own lang-goal preprocessing) -- lets us
    exercise the real batching logic without mdt_policy/a checkpoint."""

    def __init__(self, act_window_size=4, embed_dim=3):
        self.use_text_not_embedding = False
        self.act_window_size = act_window_size
        self.embed_dim = embed_dim
        self.compute_voltron_embeddings_calls = 0

    def language_goal(self, lang):
        return torch.full((self.embed_dim,), float(lang), dtype=torch.float32)

    def compute_voltron_embeddings(self, rgb_static, rgb_gripper):
        del rgb_gripper
        self.compute_voltron_embeddings_calls += 1
        batch = rgb_static.shape[0]
        return {"state_images": torch.zeros(batch, 1, self.embed_dim)}

    def denoise_actions(self, latent_plan, perceptual_emb, latent_goal, inference=False):
        del latent_plan, inference
        batch = latent_goal.shape[0]
        # every row must have received the *same* tiled conditioning --
        # verifies the .expand()/.contiguous() tiling broadcasts correctly
        for row in latent_goal[1:]:
            assert torch.allclose(row, latent_goal[0])
        for row in perceptual_emb["state_images"][1:]:
            assert torch.allclose(row, perceptual_emb["state_images"][0])
        return torch.randn(batch, self.act_window_size, 7)


def _observation(lang=1.0):
    return {
        "goal": {"lang": lang},
        "rgb_obs": {
            "rgb_static": torch.zeros(1, 1, 3, 2, 2),
            "rgb_gripper": torch.zeros(1, 1, 3, 2, 2),
        },
    }


def test_forward_only_policy_returns_k_chunks_of_the_right_shape():
    model = _FakeMDTModel(act_window_size=4)
    policy = ForwardOnlyPolicy(model, n_candidates=5)
    candidates = policy.propose(_observation())
    assert len(candidates) == 5
    for chunk in candidates:
        assert chunk.shape == (4, 7)


def test_forward_only_policy_runs_the_vision_encoder_once_not_k_times():
    model = _FakeMDTModel()
    policy = ForwardOnlyPolicy(model, n_candidates=5)
    policy.propose(_observation())
    assert model.compute_voltron_embeddings_calls == 1


def test_forward_only_policy_yields_genuinely_distinct_candidates():
    model = _FakeMDTModel()
    policy = ForwardOnlyPolicy(model, n_candidates=5)
    candidates = policy.propose(_observation())
    assert not np.allclose(candidates[0], candidates[1])


def test_forward_only_policy_rejects_a_visual_only_goal():
    model = _FakeMDTModel()
    policy = ForwardOnlyPolicy(model, n_candidates=5)
    observation = _observation()
    del observation["goal"]["lang"]
    with pytest.raises(NotImplementedError, match="lang"):
        policy.propose(observation)
