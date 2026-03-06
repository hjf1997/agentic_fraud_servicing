"""Full end-to-end simulation: scam-disguised-as-fraud scenario with live Bedrock LLM.

Orchestrates a realistic AMEX fraud dispute call where a scammed cardmember
(John Smith) tries to frame a legitimate purchase as unauthorized fraud. The
system uses real AWS Bedrock LLM calls for everything: CCP/CM conversation
generation, copilot specialist agents, and investigator specialist agents.
Only the backend evidence data is simulated.

Usage:
    python scripts/run_simulation.py

Requires valid AWS credentials in .env (LLM_PROVIDER=bedrock).
"""

import asyncio
import json
import os
import sys
import uuid

# Ensure project root is on sys.path so 'scripts' package is importable
# when running directly via: python scripts/run_simulation.py
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from agentic_fraud_servicing.config import get_settings
from agentic_fraud_servicing.copilot.orchestrator import CopilotOrchestrator
from agentic_fraud_servicing.ingestion.transcript import parse_transcript_event
from agentic_fraud_servicing.investigator.orchestrator import InvestigatorOrchestrator
from agentic_fraud_servicing.models.case import CopilotSuggestion
from agentic_fraud_servicing.providers.base import get_model_provider
from agentic_fraud_servicing.ui.helpers import create_gateway
from scripts.simulation_data import (
    create_initial_case,
    generate_ccp_turn,
    generate_cm_turn,
    seed_evidence,
)

# ---------------------------------------------------------------------------
# ANSI color codes for terminal output
# ---------------------------------------------------------------------------

BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
DIM = "\033[2m"
RESET = "\033[0m"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CASE_ID = "case-sim-e2e-001"
CALL_ID = "call-sim-e2e-001"
DB_DIR = "data/simulation"
MAX_TURNS = 14

# System events injected at specific turns
SYSTEM_EVENT_AUTH = (
    "SYSTEM: Identity verification complete. Caller confirmed as John Smith, "
    "AMEX card ending in 0005. Account in good standing, no prior disputes."
)
SYSTEM_EVENT_EVIDENCE = (
    "SYSTEM: Transaction details retrieved — $2,847.99 at TechVault Electronics, "
    "7 days ago. Authentication: chip+PIN on enrolled device (dev-js-enrolled-001). "
    "Delivery: signed for at cardholder address 6 days ago (tracking TRACK-TV-78901)."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(call_id: str, turn: int, speaker: str, text: str) -> dict:
    """Build a raw transcript event dict for parse_transcript_event."""
    return {
        "call_id": call_id,
        "event_id": f"evt-{uuid.uuid4().hex[:8]}",
        "timestamp_ms": 1000 + turn * 5000,
        "speaker": speaker,
        "text": text,
        "confidence": 1.0,
        "meta": {"channel": "phone", "locale": "en-US"},
    }


def _format_copilot_context(suggestion: CopilotSuggestion) -> str:
    """Format a CopilotSuggestion as a readable text block for the CCP agent."""
    lines = []
    scores = suggestion.hypothesis_scores
    lines.append(
        f"Hypothesis scores: FRAUD={scores.get('FRAUD', 0):.2f}, "
        f"DISPUTE={scores.get('DISPUTE', 0):.2f}, "
        f"SCAM={scores.get('SCAM', 0):.2f}"
    )
    lines.append(f"Impersonation risk: {suggestion.impersonation_risk:.2f}")
    if suggestion.risk_flags:
        lines.append(f"Risk flags: {'; '.join(suggestion.risk_flags[:5])}")
    if suggestion.suggested_questions:
        lines.append("Suggested questions:")
        for q in suggestion.suggested_questions[:3]:
            lines.append(f"  - {q}")
    if suggestion.running_summary:
        lines.append(f"Summary: {suggestion.running_summary}")
    if suggestion.retrieved_facts:
        lines.append(f"Retrieved facts: {suggestion.retrieved_facts[0][:200]}")
    return "\n".join(lines)


def _print_turn(turn: int, speaker: str, text: str) -> None:
    """Print a conversation turn with color formatting."""
    color = CYAN if speaker == "CCP" else YELLOW if speaker == "CARDMEMBER" else DIM
    print(f"\n{BOLD}[Turn {turn}]{RESET} {color}{speaker}:{RESET} {text}")


def _print_copilot_brief(suggestion: CopilotSuggestion) -> None:
    """Print a brief copilot summary after each turn."""
    scores = suggestion.hypothesis_scores
    top = max(scores, key=scores.get)  # type: ignore[arg-type]
    print(
        f"  {DIM}Copilot: {top}={scores[top]:.2f} | "
        f"Impersonation={suggestion.impersonation_risk:.2f} | "
        f"Questions={len(suggestion.suggested_questions)}{RESET}"
    )


def _print_header(text: str) -> None:
    """Print a section header."""
    print(f"\n{BOLD}{GREEN}{'=' * 60}{RESET}")
    print(f"{BOLD}{GREEN}{text}{RESET}")
    print(f"{BOLD}{GREEN}{'=' * 60}{RESET}")


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------


async def main() -> None:
    """Run the full end-to-end simulation."""

    # -- Banner --
    print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  AMEX Fraud Servicing — Full E2E Simulation{RESET}")
    print(f"{BOLD}{CYAN}  (Live Bedrock LLM){RESET}")
    print(f"{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"\n{DIM}Scenario: John Smith, scammed into a $2,847.99 purchase at")
    print("TechVault Electronics, calls AMEX claiming unauthorized fraud.")
    print(f"System has chip+PIN auth, signed delivery, and legitimate merchant.{RESET}")

    # -- Setup --
    try:
        settings = get_settings()
    except Exception as exc:
        print(f"\n{RED}Error loading settings: {exc}{RESET}")
        print(f"{RED}Ensure .env is configured with LLM_PROVIDER, AWS_PROFILE, etc.{RESET}")
        sys.exit(1)

    if settings.llm_provider != "bedrock":
        print(
            f"\n{YELLOW}Warning: LLM_PROVIDER is '{settings.llm_provider}', "
            f"not 'bedrock'. Proceeding anyway.{RESET}"
        )

    try:
        model_provider = get_model_provider(settings)
    except Exception as exc:
        print(f"\n{RED}Error creating model provider: {exc}{RESET}")
        print(f"{RED}Check AWS credentials and Bedrock model configuration.{RESET}")
        sys.exit(1)

    gateway = create_gateway(DB_DIR)
    print(f"\n{DIM}Gateway created with SQLite stores in {DB_DIR}/{RESET}")

    # ===================================================================
    # Phase 1: Seed Evidence
    # ===================================================================
    _print_header("Phase 1: Seeding Evidence Store")

    seed_evidence(gateway, CASE_ID)
    case = create_initial_case(gateway, CASE_ID, CALL_ID)

    nodes = gateway.evidence_store.get_nodes_by_case(CASE_ID)
    edges = gateway.evidence_store.get_edges_by_case(CASE_ID)
    print(f"  Seeded {len(nodes)} evidence nodes, {len(edges)} edges")
    print(f"  Case {CASE_ID}: status={case.status.value}, allegation={case.allegation_type.value}")

    # ===================================================================
    # Phase 2: Copilot — Live Call Simulation
    # ===================================================================
    _print_header("Phase 2: Copilot — Live Call Simulation via Bedrock")

    copilot = CopilotOrchestrator(gateway, model_provider)
    conversation_history: list[tuple[str, str]] = []
    last_suggestion = None
    turn = 0

    # -- Turn 1: CCP greeting (no copilot context yet) --
    turn += 1
    try:
        ccp_text = await generate_ccp_turn(
            conversation_history="[Call begins. Greet the caller.]",
            copilot_context="No copilot data yet — this is the start of the call.",
            model_provider=model_provider,
        )
    except Exception as exc:
        print(f"\n{RED}CCP generation failed on turn {turn}: {exc}{RESET}")
        sys.exit(1)

    conversation_history.append(("CCP", ccp_text))
    _print_turn(turn, "CCP", ccp_text)

    raw_event = _make_event(CALL_ID, turn, "CCP", ccp_text)
    event = parse_transcript_event(raw_event)
    last_suggestion = await copilot.process_event(event)
    _print_copilot_brief(last_suggestion)

    # -- Conversation loop --
    while turn < MAX_TURNS:
        # CM turn
        turn += 1
        history_text = "\n".join(f"{spk}: {txt}" for spk, txt in conversation_history)
        try:
            cm_text = await generate_cm_turn(history_text, model_provider)
        except Exception as exc:
            print(f"\n{RED}CM generation failed on turn {turn}: {exc}{RESET}")
            break

        conversation_history.append(("CARDMEMBER", cm_text))
        _print_turn(turn, "CARDMEMBER", cm_text)

        raw_event = _make_event(CALL_ID, turn, "CARDMEMBER", cm_text)
        event = parse_transcript_event(raw_event)
        last_suggestion = await copilot.process_event(event)
        _print_copilot_brief(last_suggestion)

        # Inject SYSTEM events at key points
        if turn == 4:
            turn += 1
            conversation_history.append(("SYSTEM", SYSTEM_EVENT_AUTH))
            _print_turn(turn, "SYSTEM", SYSTEM_EVENT_AUTH)
            raw_event = _make_event(CALL_ID, turn, "SYSTEM", SYSTEM_EVENT_AUTH)
            event = parse_transcript_event(raw_event)
            last_suggestion = await copilot.process_event(event)
            _print_copilot_brief(last_suggestion)

        if turn == 10:
            turn += 1
            conversation_history.append(("SYSTEM", SYSTEM_EVENT_EVIDENCE))
            _print_turn(turn, "SYSTEM", SYSTEM_EVENT_EVIDENCE)
            raw_event = _make_event(CALL_ID, turn, "SYSTEM", SYSTEM_EVENT_EVIDENCE)
            event = parse_transcript_event(raw_event)
            last_suggestion = await copilot.process_event(event)
            _print_copilot_brief(last_suggestion)

        # Check if we have enough turns left for a CCP response
        if turn >= MAX_TURNS:
            break

        # CCP turn (with copilot context)
        turn += 1
        copilot_ctx = _format_copilot_context(last_suggestion)
        history_text = "\n".join(f"{spk}: {txt}" for spk, txt in conversation_history)
        try:
            ccp_text = await generate_ccp_turn(history_text, copilot_ctx, model_provider)
        except Exception as exc:
            print(f"\n{RED}CCP generation failed on turn {turn}: {exc}{RESET}")
            break

        conversation_history.append(("CCP", ccp_text))
        _print_turn(turn, "CCP", ccp_text)

        raw_event = _make_event(CALL_ID, turn, "CCP", ccp_text)
        event = parse_transcript_event(raw_event)
        last_suggestion = await copilot.process_event(event)
        _print_copilot_brief(last_suggestion)

    # -- Final copilot state --
    print(f"\n{BOLD}Final Copilot State:{RESET}")
    print(f"  Hypothesis scores: {json.dumps(copilot.hypothesis_scores, indent=2)}")
    print(f"  Impersonation risk: {copilot.impersonation_risk:.2f}")
    print(f"  Missing fields: {copilot.missing_fields}")
    print(f"  Evidence collected: {copilot.evidence_collected}")
    print(f"  Transcript events processed: {len(copilot.transcript_history)}")

    # ===================================================================
    # Phase 3: Investigator — Post-Call Analysis
    # ===================================================================
    _print_header("Phase 3: Investigator — Post-Call Analysis via Bedrock")

    investigator = InvestigatorOrchestrator(gateway, model_provider)
    try:
        case_pack = await investigator.investigate(CASE_ID)
    except Exception as exc:
        print(f"\n{RED}Investigation failed: {exc}{RESET}")
        print(f"{DIM}Continuing to verification phase...{RESET}")
        case_pack = None

    if case_pack is not None:
        print(f"\n{BOLD}Case Summary:{RESET}")
        print(f"  {case_pack.case_summary[:500]}")

        print(f"\n{BOLD}Timeline ({len(case_pack.timeline)} events):{RESET}")
        for entry in case_pack.timeline[:10]:
            ts = entry.get("timestamp", "?")
            desc = entry.get("description", "?")
            src = entry.get("source", "?")
            print(f"  [{ts}] {desc} ({src})")

        print(f"\n{BOLD}Evidence List ({len(case_pack.evidence_list)} items):{RESET}")
        for item in case_pack.evidence_list[:10]:
            ntype = item.get("node_type", "?")
            stype = item.get("source_type", "?")
            summary = item.get("summary", "?")
            print(f"  [{stype}] {ntype}: {summary[:80]}")

        print(f"\n{BOLD}Decision Recommendation:{RESET}")
        rec = case_pack.decision_recommendation
        print(f"  {json.dumps(rec, indent=2, default=str)}")

        if case_pack.investigation_notes:
            print(f"\n{BOLD}Investigation Notes:{RESET}")
            for note in case_pack.investigation_notes[:5]:
                print(f"  - {note}")

    # ===================================================================
    # Phase 4: Post-Simulation Verification
    # ===================================================================
    _print_header("Phase 4: Post-Simulation Verification")

    # Check case status
    final_case = gateway.case_store.get_case(CASE_ID)
    if final_case:
        print(f"  Case status: {final_case.status.value}")
    else:
        print(f"  {RED}Case not found in store!{RESET}")

    # Count evidence nodes and edges
    final_nodes = gateway.evidence_store.get_nodes_by_case(CASE_ID)
    final_edges = gateway.evidence_store.get_edges_by_case(CASE_ID)
    print(f"  Evidence nodes: {len(final_nodes)} (started with {len(nodes)})")
    print(f"  Evidence edges: {len(final_edges)} (started with {len(edges)})")

    # Check for InvestigatorNote
    inv_notes = [n for n in final_nodes if n.get("node_type") == "INVESTIGATOR_NOTE"]
    print(f"  InvestigatorNote nodes: {len(inv_notes)}")

    # Count trace records
    traces = gateway.trace_store.get_traces_by_case(CASE_ID)
    print(f"  Trace records: {len(traces)}")

    # Summary
    print(f"\n{BOLD}{GREEN}Simulation complete.{RESET}")
    all_ok = final_case is not None and len(final_nodes) > len(nodes) and len(traces) > 0
    if all_ok:
        print(f"{GREEN}All verification checks passed.{RESET}")
    else:
        print(f"{YELLOW}Some verification checks may have issues — review output above.{RESET}")


if __name__ == "__main__":
    asyncio.run(main())
