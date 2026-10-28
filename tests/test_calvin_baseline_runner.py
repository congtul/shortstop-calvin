from shortstop.calvin.baseline_runner import (
    clearance_stats, fallback_rate, intervention_precision, latency_stats, progress_latency_stats,
    propose_latency_stats, shield_activation_rate,
)


def _attempt(
    violated=False, reached=False, min_clearance=None, n_decisions=1, n_activated=0, n_fallback=0,
    latencies_ms=None, rejected_total=0, rejected_truly_unsafe=0, propose_latencies_ms=None,
    progress_latencies_ms=None,
):
    return {
        "violated": violated, "reached": reached, "min_clearance": min_clearance,
        "n_decisions": n_decisions, "n_activated": n_activated, "n_fallback": n_fallback,
        "latencies_ms": latencies_ms or [], "rejected_total": rejected_total,
        "rejected_truly_unsafe": rejected_truly_unsafe,
        "propose_latencies_ms": propose_latencies_ms or [],
        "progress_latencies_ms": progress_latencies_ms or [],
    }


def test_clearance_stats_ignores_attempts_with_no_obstacle():
    sequence_results = [[_attempt(min_clearance=0.1), _attempt(min_clearance=None), _attempt(min_clearance=-0.2)]]
    stats = clearance_stats(sequence_results)
    assert stats["n"] == 2
    assert stats["min"] == -0.2
    assert stats["max"] == 0.1


def test_clearance_stats_none_when_nothing_measured():
    assert clearance_stats([[_attempt(min_clearance=None)]]) is None


def test_shield_activation_rate_is_pooled_not_averaged():
    sequence_results = [
        [_attempt(n_decisions=10, n_activated=1)],
        [_attempt(n_decisions=1, n_activated=1)],
    ]
    # pooled: 2/11, NOT the mean of per-attempt rates (0.1 and 1.0 -> 0.55)
    assert shield_activation_rate(sequence_results) == 2 / 11


def test_shield_activation_rate_none_when_no_decisions():
    assert shield_activation_rate([]) is None


def test_fallback_rate_distinct_from_activation_rate():
    """A decision can be 'activated' (partial rejection) without being a
    total fallback -- fallback_rate must only count n_fallback, and must
    stay well-defined (default 0) for attempts that lack the n_fallback
    field entirely."""
    sequence_results = [[_attempt(n_decisions=10, n_activated=4, n_fallback=1)]]
    assert shield_activation_rate(sequence_results) == 0.4
    assert fallback_rate(sequence_results) == 0.1


def test_fallback_rate_defaults_to_zero_for_legacy_attempts_missing_the_field():
    legacy_attempt = {"n_decisions": 5, "n_activated": 2}  # no "n_fallback" key at all
    assert fallback_rate([[legacy_attempt]]) == 0.0


def test_latency_stats_pools_every_decisions_latency():
    sequence_results = [
        [_attempt(latencies_ms=[1.0, 2.0])],
        [_attempt(latencies_ms=[3.0])],
    ]
    stats = latency_stats(sequence_results)
    assert stats["n"] == 3
    assert stats["mean"] == 2.0


def test_latency_stats_none_when_nothing_recorded():
    assert latency_stats([[_attempt(latencies_ms=[])]]) is None


def test_propose_latency_stats_pools_every_decisions_latency():
    sequence_results = [
        [_attempt(propose_latencies_ms=[10.0, 20.0])],
        [_attempt(propose_latencies_ms=[30.0])],
    ]
    stats = propose_latency_stats(sequence_results)
    assert stats["n"] == 3
    assert stats["mean"] == 20.0


def test_propose_latency_stats_none_when_nothing_recorded():
    assert propose_latency_stats([[_attempt(propose_latencies_ms=[])]]) is None


def test_propose_latency_stats_defaults_to_empty_for_legacy_attempts_missing_the_field():
    legacy_attempt = {"violated": False, "reached": True}  # no "propose_latencies_ms" key at all
    assert propose_latency_stats([[legacy_attempt]]) is None


def test_progress_latency_stats_pools_every_decisions_latency():
    sequence_results = [
        [_attempt(progress_latencies_ms=[1.0, 3.0])],
        [_attempt(progress_latencies_ms=[5.0])],
    ]
    stats = progress_latency_stats(sequence_results)
    assert stats["n"] == 3
    assert stats["mean"] == 3.0


def test_progress_latency_stats_none_for_unshielded_attempts_missing_the_field():
    """run_calvin_unshielded_subtask never computes g(a) -- its attempts
    have no 'progress_latencies_ms' key at all."""
    unshielded_attempt = {"violated": False, "reached": True}
    assert progress_latency_stats([[unshielded_attempt]]) is None


def test_intervention_precision_pools_rejections():
    sequence_results = [
        [_attempt(rejected_total=4, rejected_truly_unsafe=1)],
        [_attempt(rejected_total=1, rejected_truly_unsafe=1)],
    ]
    assert intervention_precision(sequence_results) == 2 / 5


def test_intervention_precision_none_when_nothing_rejected():
    assert intervention_precision([[_attempt(rejected_total=0, rejected_truly_unsafe=0)]]) is None
