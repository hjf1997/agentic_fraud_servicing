"""Latency evaluator — pure Python analysis of per-turn copilot response times.

Computes percentile distribution (p50/p95/p99/max), compliance rate against a
1500ms target, and flags turns that exceed the threshold.
"""

from __future__ import annotations

from agentic_fraud_servicing.evaluation.models import EvaluationRun, LatencyReport


def _percentile(sorted_values: list[float], p: float) -> float:
    """Compute the p-th percentile from a pre-sorted list using nearest-rank.

    Args:
        sorted_values: Non-empty list of floats in ascending order.
        p: Percentile in [0, 100].

    Returns:
        The value at the given percentile.
    """
    n = len(sorted_values)
    # Nearest-rank method: index = ceil(p/100 * n) - 1, clamped to valid range
    idx = int(p / 100.0 * n + 0.5) - 1
    idx = max(0, min(idx, n - 1))
    return sorted_values[idx]


def evaluate_latency(run: EvaluationRun) -> LatencyReport:
    """Analyze per-turn copilot latency from an evaluation run.

    Args:
        run: A completed EvaluationRun containing turn_metrics with latency_ms.

    Returns:
        LatencyReport with percentiles, compliance rate, and flagged turns.
    """
    # Only include assessed turns (those with a copilot_suggestion) so that
    # CCP/SYSTEM turns with near-zero latency don't dilute the statistics.
    assessed = [tm for tm in run.turn_metrics if tm.copilot_suggestion is not None]

    if not assessed:
        return LatencyReport(
            per_turn_latency_ms=[],
            p50_ms=0.0,
            p95_ms=0.0,
            p99_ms=0.0,
            max_ms=0.0,
            compliance_rate=0.0,
            flagged_turns=[],
        )

    latencies = [tm.latency_ms for tm in assessed]
    sorted_latencies = sorted(latencies)

    target = 1500.0
    compliant_count = sum(1 for lat in latencies if lat <= target)
    compliance_rate = compliant_count / len(latencies)

    flagged_turns = [tm.turn_number for tm in assessed if tm.latency_ms > target]

    return LatencyReport(
        per_turn_latency_ms=latencies,
        p50_ms=_percentile(sorted_latencies, 50),
        p95_ms=_percentile(sorted_latencies, 95),
        p99_ms=_percentile(sorted_latencies, 99),
        max_ms=sorted_latencies[-1],
        compliance_rate=compliance_rate,
        flagged_turns=flagged_turns,
        assessed_turns=[tm.turn_number for tm in assessed],
    )
