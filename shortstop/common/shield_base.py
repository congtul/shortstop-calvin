"""Mask -> argmax-or-fallback bookkeeping shared by the 2D and CALVIN shields.

Subclasses define admissibility; `_select_by_mask` returns the admissible
candidate with the best score (the first one on ties) or, if none is
admissible, a zero chunk with info["fallback"] = True. CALVIN shields take
caller-supplied scores (g(a)); the 2D ones compute -goal_distance themselves.
Repair and QP shields keep their own select() and only reuse this last step.
"""
import numpy as np


class ShieldBase:
    def _admissible(self, state, chunk):
        raise NotImplementedError

    def _score(self, state, chunk, scores=None, index=None):
        """scores[index]; the 2D shields override this."""
        if scores is None:
            raise NotImplementedError("caller-supplied scores required")
        return scores[index]

    def _fallback(self, candidates):
        return np.zeros_like(candidates[0])

    def _select_by_mask(self, state, candidates, mask, scores=None, extra_info=None):
        admissible_idx = [i for i, ok in enumerate(mask) if ok]
        extra = extra_info or {}
        if not admissible_idx:
            return self._fallback(candidates), {
                "fallback": True, "n_admissible": 0, "admissible_mask": mask, **extra,
            }
        best_i = max(admissible_idx, key=lambda i: self._score(state, candidates[i], scores, i))
        return candidates[best_i], {
            "fallback": False, "n_admissible": len(admissible_idx), "admissible_mask": mask, **extra,
        }

    def select(self, state, candidates, scores=None):
        """Mask every candidate with `_admissible`, then `_select_by_mask`."""
        mask = [self._admissible(state, c) for c in candidates]
        return self._select_by_mask(state, candidates, mask, scores)
