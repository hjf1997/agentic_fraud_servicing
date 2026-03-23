"""Evidence utilization evaluator — pure Python analysis of evidence coverage.

Measures how much of the available evidence was retrieved and referenced in
the copilot's reasoning during an evaluation run.
"""

from __future__ import annotations

from agentic_fraud_servicing.evaluation.models import EvaluationRun, EvidenceUtilizationResult


def evaluate_evidence_utilization(run: EvaluationRun) -> EvidenceUtilizationResult:
    """Analyze evidence retrieval and reasoning coverage from an evaluation run.

    Compares key_evidence_nodes from ground truth against retrieved_facts and
    running_summary/risk_flags in copilot suggestions.

    Args:
        run: A completed EvaluationRun with turn_metrics containing copilot_suggestion.

    Returns:
        EvidenceUtilizationResult with coverage rates and missed evidence list.
    """
    key_nodes: list[str] = run.ground_truth.get("key_evidence_nodes", [])
    total = len(key_nodes)

    if total == 0:
        return EvidenceUtilizationResult(
            total_evidence_nodes=0,
            retrieved_nodes=0,
            referenced_in_reasoning=0,
            retrieval_coverage=0.0,
            reasoning_coverage=0.0,
            missed_evidence=[],
        )

    # Collect all unique retrieved fact references across all turns
    retrieved_ids: set[str] = set()
    reasoning_text_parts: list[str] = []

    for tm in run.turn_metrics:
        suggestion = tm.copilot_suggestion
        if not suggestion:
            continue

        # Gather retrieved_facts entries
        facts = suggestion.get("retrieved_facts", [])
        for fact in facts:
            if isinstance(fact, dict):
                # Look for node_id or id keys in fact dicts
                node_id = fact.get("node_id") or fact.get("id") or ""
                if node_id:
                    retrieved_ids.add(str(node_id))
            elif isinstance(fact, str):
                retrieved_ids.add(fact)

        # Gather text for reasoning reference check
        summary = suggestion.get("running_summary", "")
        if summary:
            reasoning_text_parts.append(str(summary))
        risk_flags = suggestion.get("risk_flags", [])
        for flag in risk_flags:
            reasoning_text_parts.append(str(flag))

    # Combine all reasoning text for substring matching
    reasoning_text = " ".join(reasoning_text_parts)

    # Count how many key nodes appear in retrieved facts
    retrieved_count = sum(1 for nid in key_nodes if nid in retrieved_ids)

    # Count how many key nodes are referenced in reasoning text
    referenced_count = sum(1 for nid in key_nodes if nid in reasoning_text)

    # Build missed evidence list: nodes not found in either retrieval or reasoning
    missed = [
        {"node_id": nid}
        for nid in key_nodes
        if nid not in retrieved_ids and nid not in reasoning_text
    ]

    return EvidenceUtilizationResult(
        total_evidence_nodes=total,
        retrieved_nodes=retrieved_count,
        referenced_in_reasoning=referenced_count,
        retrieval_coverage=retrieved_count / total,
        reasoning_coverage=referenced_count / total,
        missed_evidence=missed,
    )
