"""Propose step for CALVIN: K candidate action chunks from the MDT policy.

MDT's own `step()` samples a (10, 7) chunk, caches it and returns one action
per call, so the chunk is never visible. `ForwardOnlyPolicy` instead runs
the model's language-goal preprocessing once and calls `denoise_actions`
with the conditioning repeated K times: the diffusion noise is drawn per
batch row, so the K chunks are independent samples from one batched pass.
Columns: 3 TCP position deltas, 3 Euler deltas, gripper (CALVIN's relative
action space).
"""
import numpy as np
import torch


class ForwardOnlyPolicy:
    """`propose(obs)` -> K (10, 7) chunks from an already loaded and
    configured MDTVAgent. Language goals only. The preprocessing copies
    MDTVAgent.forward's (which hard-codes batch 1), then expands the
    conditioning to batch K."""

    def __init__(self, model, n_candidates=1):
        self.model = model
        self.n_candidates = n_candidates

    def propose(self, observation):
        goal = observation["goal"]
        model = self.model
        k = self.n_candidates

        if "lang" not in goal:
            raise NotImplementedError(
                "ForwardOnlyPolicy only supports language goals (goal={'lang': ...})"
            )

        rgb_static = observation["rgb_obs"]["rgb_static"]
        rgb_gripper = observation["rgb_obs"]["rgb_gripper"]

        # MDTVAgent.forward's lang-goal preprocessing, batch 1.
        if model.use_text_not_embedding:
            latent_goal = model.language_goal(goal["lang_text"])
            latent_goal = latent_goal.to(torch.float32)
        else:
            latent_goal = model.language_goal(goal["lang"]).unsqueeze(0).to(torch.float32).to(rgb_static.device)

        perceptual_emb = model.compute_voltron_embeddings(rgb_static, rgb_gripper)
        perceptual_emb["modality"] = "lang"

        # batch 1 -> k (contiguous, not a stride-0 view).
        latent_goal_k = latent_goal.expand(k, *latent_goal.shape[1:]).contiguous()
        perceptual_emb_k = {
            key: (value.expand(k, *value.shape[1:]).contiguous() if torch.is_tensor(value) else value)
            for key, value in perceptual_emb.items()
        }
        latent_plan_k = torch.zeros_like(latent_goal_k)

        action_seq = model.denoise_actions(latent_plan_k, perceptual_emb_k, latent_goal_k, inference=True)
        return [np.asarray(action_seq[i].detach().cpu()) for i in range(k)]


class MockMDTPolicyClient:
    """Random-chunk stand-in with the same `propose` interface, for tests."""

    def __init__(self, horizon=10, action_dim=7, n_candidates=8, noise_std=0.02, rng=None):
        self.horizon = horizon
        self.action_dim = action_dim
        self.n_candidates = n_candidates
        self.noise_std = noise_std
        self.rng = rng if rng is not None else np.random.default_rng()

    def propose(self, observation):
        del observation
        return [
            self.rng.normal(0.0, self.noise_std, size=(self.horizon, self.action_dim))
            for _ in range(self.n_candidates)
        ]
