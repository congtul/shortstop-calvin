import numpy as np

from shortstop.calvin.metrics import (
    attempted_only, bootstrap_ci, build_fixed_cohort_slots, conservatism_cost, fixed_cohort_rates, recovery_rate,
)


def _ok(n):
    return [{"violated": False, "reached": True} for _ in range(n)]


def test_build_fixed_cohort_slots_pads_truncated_sequences_with_untried_placeholders():
    sequence_results = [_ok(5), _ok(3), []]  # full success, fail at #4, fail at #1
    slots = build_fixed_cohort_slots(sequence_results, subtasks_per_sequence=5)

    assert len(slots) == 15  # 3 sequences * 5, fixed regardless of where each stopped
    assert sum(s["attempted"] for s in slots) == 5 + 3 + 0
    # untried slots (beyond where a sequence stopped) count as neither reached nor violated
    untried = [s for s in slots if not s["attempted"]]
    assert len(untried) == 15 - 8
    assert all(not s["reached"] and not s["violated"] for s in untried)


def test_build_fixed_cohort_slots_rejects_more_attempts_than_the_cap():
    import pytest
    with pytest.raises(ValueError):
        build_fixed_cohort_slots([_ok(6)], subtasks_per_sequence=5)


def test_success_rate_equals_avg_seq_len_over_five():
    # sequences completing 5, 4, 3 subtasks -> avg_seq_len = 4.0 -> success_rate = 4.0/5
    sequence_results = [_ok(5), _ok(4) + [{"violated": False, "reached": False}], _ok(3) + [{"violated": False, "reached": False}] * 2]
    slots = build_fixed_cohort_slots(sequence_results, subtasks_per_sequence=5)
    _, success_rate = fixed_cohort_rates(slots)
    assert abs(success_rate - 4.0 / 5) < 1e-9


def test_obstacle_present_raises_violation_rate_and_lowers_success_rate_vs_absent():
    # Same underlying policy behavior; "without obstacle" -> nothing ever
    # flagged, subtasks run to natural completion. "With obstacle" -> the
    # 2nd subtask of 2 of the 3 sequences gets flagged mid-way, truncating
    # that sequence right there (violated=True implies reached=False for
    # that attempt, and every later subtask in that sequence is untried).
    without_obstacle = [_ok(5), _ok(5), _ok(5)]

    with_obstacle = [
        _ok(5),
        [{"violated": False, "reached": True}, {"violated": True, "reached": False}],
        [{"violated": False, "reached": True}, {"violated": True, "reached": False}],
    ]

    slots_without = build_fixed_cohort_slots(without_obstacle)
    slots_with = build_fixed_cohort_slots(with_obstacle)

    violation_without, success_without = fixed_cohort_rates(slots_without)
    violation_with, success_with = fixed_cohort_rates(slots_with)

    assert violation_without == 0.0
    assert success_without == 1.0
    assert violation_with > violation_without
    assert success_with < success_without


def test_attempted_only_drops_untried_slots():
    slots = build_fixed_cohort_slots([_ok(2)], subtasks_per_sequence=5)
    assert len(attempted_only(slots)) == 2


def test_recovery_rate_only_counts_activated_attempts_and_ignores_untouched_ones():
    sequence_results = [[
        {"n_activated": 0, "reached": False},  # never activated -- excluded from the rate
        {"n_activated": 2, "reached": True},   # activated, task still completed -- a "recovery"
        {"n_activated": 1, "reached": False},  # activated, task did NOT complete -- not a recovery
    ]]
    # 2 activated attempts, 1 of them reached -> rate = 0.5
    assert recovery_rate(sequence_results) == 0.5


def test_recovery_rate_is_none_when_the_shield_never_activated_at_all():
    sequence_results = [[{"n_activated": 0, "reached": True}, {"n_activated": 0, "reached": True}]]
    assert recovery_rate(sequence_results) is None


def test_conservatism_cost_only_compares_paired_benign_subtasks():
    # sequence 0: unshielded benign both subtasks, shielded succeeds only
    # the first (a real conservatism cost); sequence 1: unshielded's first
    # subtask violated -- excluded from the comparison entirely, whatever
    # the shielded run did there.
    unshielded = [
        [{"violated": False, "reached": True}, {"violated": False, "reached": True}],
        [{"violated": True, "reached": False}],
    ]
    shielded = [
        [{"violated": False, "reached": True}, {"violated": False, "reached": False}],
        [{"violated": False, "reached": True}],  # would-be "improvement" on a non-benign subtask -- ignored
    ]
    # paired benign subtasks: seq0#0 (1.0 vs 1.0), seq0#1 (1.0 vs 0.0) -> unshielded=1.0, shielded=0.5
    assert abs(conservatism_cost(unshielded, shielded) - 0.5) < 1e-9


def test_conservatism_cost_is_none_when_nothing_was_ever_benign():
    unshielded = [[{"violated": True, "reached": False}]]
    shielded = [[{"violated": False, "reached": False}]]
    assert conservatism_cost(unshielded, shielded) is None


def _violation_rate_stat(sequence_results):
    return fixed_cohort_rates(build_fixed_cohort_slots(sequence_results))[0]


def test_bootstrap_ci_mean_matches_the_point_estimate_when_every_sequence_is_identical():
    # No real variability across sequences -- every resample is the exact
    # same violation_rate, so mean must equal it exactly and std must be 0.
    sequence_results = [[{"violated": False, "reached": True}]] * 20
    mean, std, values = bootstrap_ci(sequence_results, _violation_rate_stat, n_resamples=500, seed=0)
    assert abs(mean - 0.0) < 1e-9
    assert std == 0.0
    assert len(values) == 500


def test_bootstrap_ci_mean_is_close_to_the_point_estimate_with_real_variability():
    # 10 sequences violated, 10 not -> point estimate violation_rate=0.5;
    # bootstrap mean over many resamples should land close to that (law of
    # large numbers), and std must be strictly positive (real resample-to-
    # resample variability, unlike the identical-sequences case above).
    violated = [{"violated": True, "reached": False}]
    clean = [{"violated": False, "reached": True}]
    sequence_results = [violated] * 10 + [clean] * 10
    point_estimate = _violation_rate_stat(sequence_results)
    mean, std, values = bootstrap_ci(sequence_results, _violation_rate_stat, n_resamples=2000, seed=0)
    assert abs(mean - point_estimate) < 0.05
    assert std > 0.0
    assert values.shape == (2000,)


def test_bootstrap_ci_is_reproducible_given_the_same_seed():
    sequence_results = [[{"violated": True, "reached": False}]] * 5 + [[{"violated": False, "reached": True}]] * 5
    mean_a, std_a, values_a = bootstrap_ci(sequence_results, _violation_rate_stat, n_resamples=300, seed=42)
    mean_b, std_b, values_b = bootstrap_ci(sequence_results, _violation_rate_stat, n_resamples=300, seed=42)
    assert mean_a == mean_b
    assert std_a == std_b
    assert np.array_equal(values_a, values_b)
