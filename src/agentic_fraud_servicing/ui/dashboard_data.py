"""Dashboard data loader — reads simulation results from SQLite stores.

Pure data-loading module with no Gradio dependency. Each function opens its
own store connection, reads the data, and closes the connection. All functions
handle missing files and empty databases gracefully.
"""

import json
import os
import sqlite3

from agentic_fraud_servicing.storage.case_store import CaseStore
from agentic_fraud_servicing.storage.evidence_store import EvidenceStore
from agentic_fraud_servicing.storage.trace_store import TraceStore


def discover_scenarios(base_dir: str = "data/simulation") -> list[str]:
    """Scan base_dir for subdirectories containing a cases.db file.

    Args:
        base_dir: Root directory to scan for scenario subdirectories.

    Returns:
        Sorted list of scenario names (directory names). Empty if base_dir
        doesn't exist or contains no valid scenarios.
    """
    if not os.path.isdir(base_dir):
        return []

    scenarios = []
    for entry in os.listdir(base_dir):
        scenario_dir = os.path.join(base_dir, entry)
        if os.path.isdir(scenario_dir) and os.path.isfile(os.path.join(scenario_dir, "cases.db")):
            scenarios.append(entry)

    return sorted(scenarios)


def load_case(db_dir: str) -> dict | None:
    """Load the first case from cases.db as a dict.

    Opens CaseStore, queries for all cases via raw SQL (since CaseStore only
    exposes status-filtered queries), returns the first result as a dict.

    Args:
        db_dir: Directory containing cases.db.

    Returns:
        Case dict via model_dump(mode='json'), or None if no cases found.
    """
    db_path = os.path.join(db_dir, "cases.db")
    if not os.path.isfile(db_path):
        return None

    store = CaseStore(db_path)
    try:
        # CaseStore doesn't have a "list all" method, so query directly
        row = store._conn.execute(
            "SELECT data FROM cases ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])
    except (sqlite3.Error, json.JSONDecodeError):
        return None
    finally:
        store.close()


def load_transcript_turns(db_dir: str, case_id: str) -> list[dict]:
    """Load conversation transcript turns from the trace store.

    Filters traces for agent_id='transcript' and action='conversation_turn'.
    Parses output_data JSON to extract turn number, speaker, and text.

    Args:
        db_dir: Directory containing traces.db.
        case_id: The case to load transcript for.

    Returns:
        List of dicts sorted by turn: [{'turn': int, 'speaker': str, 'text': str}].
        Empty list if no transcript traces found.
    """
    db_path = os.path.join(db_dir, "traces.db")
    if not os.path.isfile(db_path):
        return []

    store = TraceStore(db_path)
    try:
        traces = store.get_traces_by_case(case_id)
    finally:
        store.close()

    turns = []
    for trace in traces:
        if trace.get("agent_id") == "transcript" and trace.get("action") == "conversation_turn":
            try:
                output = json.loads(trace["output_data"])
                turns.append(
                    {
                        "turn": output.get("turn", 0),
                        "speaker": output.get("speaker", ""),
                        "text": output.get("text", ""),
                    }
                )
            except (json.JSONDecodeError, KeyError):
                continue

    turns.sort(key=lambda t: t["turn"])
    return turns


def load_copilot_suggestions(db_dir: str, case_id: str) -> list[dict]:
    """Load copilot suggestions from the trace store.

    Filters traces for agent_id='copilot_suggestion' and action='suggestion'.
    Parses output_data as the full CopilotSuggestion dict, and extracts the
    turn number from input_data.

    Args:
        db_dir: Directory containing traces.db.
        case_id: The case to load suggestions for.

    Returns:
        List of dicts sorted by turn: [{'turn': int, 'suggestion': dict}].
        Empty list if none found.
    """
    db_path = os.path.join(db_dir, "traces.db")
    if not os.path.isfile(db_path):
        return []

    store = TraceStore(db_path)
    try:
        traces = store.get_traces_by_case(case_id)
    finally:
        store.close()

    suggestions = []
    for trace in traces:
        if trace.get("agent_id") == "copilot_suggestion" and trace.get("action") == "suggestion":
            try:
                suggestion = json.loads(trace["output_data"])
                input_data = json.loads(trace["input_data"])
                turn = input_data.get("turn", 0)
                suggestions.append({"turn": turn, "suggestion": suggestion})
            except (json.JSONDecodeError, KeyError):
                continue

    suggestions.sort(key=lambda s: s["turn"])
    return suggestions


def load_copilot_final_state(db_dir: str, case_id: str) -> dict | None:
    """Load the final copilot state from the trace store.

    Filters for agent_id='copilot_final' and action='final_state'.

    Args:
        db_dir: Directory containing traces.db.
        case_id: The case to load final state for.

    Returns:
        State dict with hypothesis_scores, impersonation_risk, etc. or None.
    """
    db_path = os.path.join(db_dir, "traces.db")
    if not os.path.isfile(db_path):
        return None

    store = TraceStore(db_path)
    try:
        traces = store.get_traces_by_case(case_id)
    finally:
        store.close()

    for trace in traces:
        if trace.get("agent_id") == "copilot_final" and trace.get("action") == "final_state":
            try:
                return json.loads(trace["output_data"])
            except (json.JSONDecodeError, KeyError):
                return None

    return None


def load_evidence(db_dir: str, case_id: str) -> tuple[list[dict], list[dict]]:
    """Load evidence nodes and edges from the evidence store.

    Args:
        db_dir: Directory containing evidence.db.
        case_id: The case to load evidence for.

    Returns:
        Tuple of (nodes, edges) where each is a list of dicts.
        Returns ([], []) if no evidence found or DB doesn't exist.
    """
    db_path = os.path.join(db_dir, "evidence.db")
    if not os.path.isfile(db_path):
        return [], []

    store = EvidenceStore(db_path)
    try:
        nodes = store.get_nodes_by_case(case_id)
        edges = store.get_edges_by_case(case_id)
    finally:
        store.close()

    return nodes, edges


def load_case_pack(db_dir: str, case_id: str) -> dict | None:
    """Load the investigator CasePack from the trace store.

    Filters for agent_id='investigator' and action='case_pack'.

    Args:
        db_dir: Directory containing traces.db.
        case_id: The case to load the case pack for.

    Returns:
        CasePack dict or None if not found.
    """
    db_path = os.path.join(db_dir, "traces.db")
    if not os.path.isfile(db_path):
        return None

    store = TraceStore(db_path)
    try:
        traces = store.get_traces_by_case(case_id)
    finally:
        store.close()

    for trace in traces:
        if trace.get("agent_id") == "investigator" and trace.get("action") == "case_pack":
            try:
                return json.loads(trace["output_data"])
            except (json.JSONDecodeError, KeyError):
                return None

    return None


def load_audit_trail(db_dir: str, case_id: str) -> list[dict]:
    """Load all trace records for a case (the full audit trail).

    Returns all traces ordered by timestamp ascending (as returned by the store).

    Args:
        db_dir: Directory containing traces.db.
        case_id: The case to load traces for.

    Returns:
        List of trace dicts. Empty list if none found or DB doesn't exist.
    """
    db_path = os.path.join(db_dir, "traces.db")
    if not os.path.isfile(db_path):
        return []

    store = TraceStore(db_path)
    try:
        return store.get_traces_by_case(case_id)
    finally:
        store.close()
