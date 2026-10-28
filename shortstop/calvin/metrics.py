"""Episode-level metrics over CALVIN rollout results.

`sequence_results`: one list per launched rollout, holding the per-subtask
results in order (CALVIN stops a sequence at the first failure). Rates use
a fixed denominator of launched slots -- a slot never reached counts as
neither success nor violation -- so a baseline that fails early is not
scored on fewer samples. With one task per sequence each rollout is one slot.
"""
import numpy as np


def build_fixed_cohort_slots(sequence_results, subtasks_per_sequence=5):
    """Flat list of len(sequence_results) * subtasks_per_sequence slots
    {violated, reached, attempted}; slots after a sequence stopped are
    unattempted."""
    slots = []
    for attempts in sequence_results:
        if len(attempts) > subtasks_per_sequence:
            raise ValueError(f"got {len(attempts)} attempts, expected at most {subtasks_per_sequence}")
        for a in attempts:
            slots.append({"violated": bool(a["violated"]), "reached": bool(a["reached"]), "attempted": True})
        for _ in range(subtasks_per_sequence - len(attempts)):
            slots.append({"violated": False, "reached": False, "attempted": False})
    return slots


def fixed_cohort_rates(slots):
    """(violation_rate, success_rate) over all slots."""
    violation_rate = float(np.mean([s["violated"] for s in slots]))
    success_rate = float(np.mean([s["reached"] for s in slots]))
    return violation_rate, success_rate


def attempted_only(slots):
    """Slots that were actually run."""
    return [s for s in slots if s["attempted"]]


def recovery_rate(sequence_results):
    """Of the attempts where the shield rejected something at least once,
    the fraction that still succeeded (paper: recovery after a rejected
    chunk, for every shield, not only repairing ones). None if never activated."""
    activated = [
        a for attempts in sequence_results for a in attempts
        if a.get("n_activated", 0) > 0
    ]
    if not activated:
        return None
    return float(np.mean([a["reached"] for a in activated]))


def conservatism_cost(unshielded_sequence_results, shielded_sequence_results):
    """Success lost to the shield on attempts whose paired unshielded run (same
    rollout index and position) did not collide: unshielded success -
    shielded success over those pairs. None if there are none."""
    benign_pairs = []
    for seq_idx, unshielded_attempts in enumerate(unshielded_sequence_results):
        shielded_attempts = shielded_sequence_results[seq_idx]
        for subtask_idx, u in enumerate(unshielded_attempts):
            if subtask_idx >= len(shielded_attempts) or u["violated"]:
                continue
            benign_pairs.append((u, shielded_attempts[subtask_idx]))
    if not benign_pairs:
        return None
    unshielded_success = np.mean([u["reached"] for u, _ in benign_pairs])
    shielded_success = np.mean([s["reached"] for _, s in benign_pairs])
    return float(unshielded_success - shielded_success)


def bootstrap_ci(sequence_results, statistic_fn, n_resamples=10_000, seed=0):
    """Bootstrap of `statistic_fn(sequence_results) -> float`, resampling whole
    rollouts with replacement `n_resamples` times; returns (mean, std,
    values). Measures the sampling noise of one run, not seed variation."""
    rng = np.random.default_rng(seed)
    n = len(sequence_results)
    values = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resampled = [sequence_results[j] for j in idx]
        values[i] = statistic_fn(resampled)
    return float(values.mean()), float(values.std()), values
