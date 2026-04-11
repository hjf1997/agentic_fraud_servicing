"""Full end-to-end simulation runner with live Bedrock LLM.

Supports multiple simulation scenarios via the --scenario flag. Each scenario
provides its own evidence data, cardmember agent, and system events. The runner
handles the shared simulation loop: CCP/CM conversation, copilot processing,
investigator analysis, and verification.

Usage:
    python scripts/run_simulation.py --scenario scam_techvault
    python scripts/run_simulation.py --scenario doordash_fraud
    python scripts/run_simulation.py --list

Requires valid AWS credentials in .env (LLM_PROVIDER=bedrock).
"""

import argparse
import asyncio
import io
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone

# Ensure project root is on sys.path so 'scripts' package is importable
# when running directly via: python scripts/run_simulation.py
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Suppress "OPENAI_API_KEY is not set, skipping trace export" noise from the
# Agents SDK when using Bedrock provider instead of OpenAI.
os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")

# Auto-discover and register all scenario_*.py modules
from agents import Agent  # noqa: E402

from scripts.simulation_data import discover_scenarios  # noqa: E402

discover_scenarios()

from agentic_fraud_servicing.config import get_settings  # noqa: E402
from agentic_fraud_servicing.copilot.orchestrator import CopilotOrchestrator  # noqa: E402
from agentic_fraud_servicing.gateway.tool_gateway import AuthContext  # noqa: E402
from agentic_fraud_servicing.gateway.tools.write_tools import (  # noqa: E402
    append_evidence_edge,
    append_evidence_node,
    mark_transactions_disputed,
)
from agentic_fraud_servicing.ingestion.redaction import redact_all  # noqa: E402
from agentic_fraud_servicing.ingestion.transcript import parse_transcript_event  # noqa: E402
from agentic_fraud_servicing.investigator.orchestrator import (
    InvestigatorOrchestrator,  # noqa: E402
)
from agentic_fraud_servicing.models.case import CopilotSuggestion  # noqa: E402
from agentic_fraud_servicing.models.enums import (  # noqa: E402
    AllegationDetailType,
    EvidenceEdgeType,
    EvidenceSourceType,
)
from agentic_fraud_servicing.models.evidence import AllegationStatement, EvidenceEdge  # noqa: E402
from agentic_fraud_servicing.providers.base import get_model_provider  # noqa: E402
from agentic_fraud_servicing.providers.bedrock_provider import BedrockModelProvider  # noqa: E402
from agentic_fraud_servicing.ui.helpers import create_gateway, load_transcript_file  # noqa: E402
from scripts.simulation_data import (  # noqa: E402
    DisputeAction,
    Scenario,
    generate_ccp_turn,
    generate_cm_turn,
    get_scenario,
    list_scenarios,
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

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")

# Haiku model for CM/CCP simulators (cheaper and faster than Sonnet)
_HAIKU_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


class TeeWriter(io.TextIOBase):
    """Duplicates writes to both the terminal and a plain-text log file.

    ANSI escape codes are stripped from the file copy so the log is readable
    in any text editor.
    """

    def __init__(self, terminal: io.TextIOBase, log_file: io.TextIOBase) -> None:
        self._terminal = terminal
        self._log_file = log_file

    def write(self, s: str) -> int:
        self._terminal.write(s)
        self._log_file.write(_ANSI_RE.sub("", s))
        return len(s)

    def flush(self) -> None:
        self._terminal.flush()
        self._log_file.flush()


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
    scores_text = ", ".join(f"{k}={v:.2f}" for k, v in scores.items())
    lines.append(f"Hypothesis scores: {scores_text}")
    if suggestion.specialist_likelihoods:
        specs_text = ", ".join(
            f"{k}={v:.2f}" for k, v in suggestion.specialist_likelihoods.items()
        )
        lines.append(f"Specialist likelihoods: {specs_text}")
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
    if suggestion.case_eligibility:
        lines.append("Case eligibility:")
        for assess in suggestion.case_eligibility:
            ctype = assess.get("case_type", "?")
            elig = assess.get("eligibility", "?")
            lines.append(f"  - {ctype}: {elig}")
            unmet = assess.get("unmet_criteria", [])
            if unmet:
                lines.append(f"    Unmet: {'; '.join(unmet[:3])}")
    if suggestion.case_advisory_summary:
        lines.append(f"Case advisory: {suggestion.case_advisory_summary}")
    return "\n".join(lines)


def _print_turn(turn: int, speaker: str, text: str) -> None:
    """Print a conversation turn with color formatting."""
    color = CYAN if speaker == "CCP" else YELLOW if speaker == "CARDMEMBER" else DIM
    print(f"\n{BOLD}[Turn {turn}]{RESET} {color}{speaker}:{RESET} {text}")


def _print_copilot_brief(suggestion: CopilotSuggestion) -> None:
    """Print full copilot suggestions after each turn."""
    scores = suggestion.hypothesis_scores
    scores_parts = " | ".join(f"{k}={v:.2f}" for k, v in scores.items())
    print(
        f"  {DIM}Hypothesis: {scores_parts} | "
        f"Impersonation={suggestion.impersonation_risk:.2f}{RESET}"
    )
    if suggestion.specialist_likelihoods:
        specs_parts = " | ".join(
            f"{k}={v:.2f}" for k, v in suggestion.specialist_likelihoods.items()
        )
        print(f"  {DIM}Specialists: {specs_parts}{RESET}")
    if suggestion.suggested_questions:
        print(f"  {CYAN}Suggested Questions:{RESET}")
        for q in suggestion.suggested_questions[:3]:
            print(f"    - {q}")
    if suggestion.information_sufficient:
        print(f"  {GREEN}Information sufficient - ready to proceed{RESET}")
    if suggestion.risk_flags:
        print(f"  {RED}Risk Flags:{RESET}")
        for flag in suggestion.risk_flags[:5]:
            print(f"    ! {flag}")
    if suggestion.running_summary:
        print(f"  {DIM}Summary: {suggestion.running_summary}{RESET}")
    if suggestion.safety_guidance:
        print(f"  {YELLOW}Safety: {suggestion.safety_guidance}{RESET}")
    if suggestion.case_eligibility:
        parts = []
        for assess in suggestion.case_eligibility:
            ctype = assess.get("case_type", "?")
            elig = assess.get("eligibility", "?")
            if elig == "eligible":
                parts.append(f"{GREEN}{ctype}={elig}{RESET}")
            elif elig == "blocked":
                parts.append(f"{RED}{ctype}={elig}{RESET}")
            else:
                parts.append(f"{YELLOW}{ctype}={elig}{RESET}")
        print(f"  Case Eligibility: {' | '.join(parts)}")


def _persist_trace(
    gateway,
    case_id: str,
    agent_id: str,
    action: str,
    input_data: str,
    output_data: str,
    duration_ms: float = 0.0,
) -> None:
    """Persist a data record to the trace store for dashboard consumption.

    Generates a unique trace_id and timestamps automatically. Conversation turn
    text is redacted before storage to prevent raw PAN/PII from reaching the DB.

    Args:
        duration_ms: Wall-clock duration of the operation in milliseconds.
            Defaults to 0.0 for pure data records (e.g. transcript turns).
    """
    # Redact PII in conversation turn text before persisting
    if action == "conversation_turn":
        try:
            data = json.loads(output_data)
            if "text" in data:
                data["text"], _ = redact_all(data["text"])
                output_data = json.dumps(data)
        except (json.JSONDecodeError, TypeError):
            pass

    gateway.trace_store.log_invocation(
        trace_id=str(uuid.uuid4()),
        case_id=case_id,
        agent_id=agent_id,
        action=action,
        input_data=input_data,
        output_data=output_data,
        duration_ms=duration_ms,
        timestamp=datetime.now(timezone.utc),
        status="success",
    )


def _print_header(text: str) -> None:
    """Print a section header."""
    print(f"\n{BOLD}{GREEN}{'=' * 60}{RESET}")
    print(f"{BOLD}{GREEN}{text}{RESET}")
    print(f"{BOLD}{GREEN}{'=' * 60}{RESET}")


async def _inject_system_event(
    turn: int,
    text: str,
    scenario: Scenario,
    copilot: "CopilotOrchestrator",
    conversation_history: list[tuple[str, str]],
    gateway,
) -> tuple[int, CopilotSuggestion | None]:
    """Inject a SYSTEM event into the conversation, process via copilot, and persist.

    Returns the updated turn number and the copilot suggestion (None on
    non-assessment turns).
    """
    turn += 1
    conversation_history.append(("SYSTEM", text))
    _print_turn(turn, "SYSTEM", text)

    raw_event = _make_event(scenario.call_id, turn, "SYSTEM", text)
    event = parse_transcript_event(raw_event)

    t0 = time.perf_counter()
    suggestion = await copilot.process_event(event)
    copilot_dur = (time.perf_counter() - t0) * 1000

    _persist_trace(
        gateway,
        scenario.case_id,
        "transcript",
        "conversation_turn",
        json.dumps({"turn": turn, "speaker": "SYSTEM"}),
        json.dumps({"turn": turn, "speaker": "SYSTEM", "text": text}),
    )
    if suggestion is not None:
        _print_copilot_brief(suggestion)
        _persist_trace(
            gateway,
            scenario.case_id,
            "copilot_suggestion",
            "suggestion",
            json.dumps({"turn": turn}),
            suggestion.model_dump_json(),
            duration_ms=copilot_dur,
        )
    return turn, suggestion


# ---------------------------------------------------------------------------
# Dispute action processing
# ---------------------------------------------------------------------------


async def _process_dispute_action(
    action: DisputeAction,
    turn: int,
    scenario: Scenario,
    copilot: "CopilotOrchestrator",
    conversation_history: list[tuple[str, str]],
    gateway,
) -> tuple[int, CopilotSuggestion]:
    """Process a CCP dispute action: create claim node, edges, and SYSTEM event.

    When a CCP identifies which transaction(s) the cardmember is disputing,
    this function:
    1. Creates an AllegationStatement evidence node (ALLEGATION) from the action's claim_text
    2. Creates EvidenceEdge(s) linking the AllegationStatement to each disputed transaction
    3. Looks up transaction details to build an informative SYSTEM event
    4. Injects the SYSTEM event into the copilot conversation

    Returns the updated turn number and the copilot suggestion produced.
    """
    ctx = AuthContext(agent_id="simulation", case_id=scenario.case_id, permissions={"write"})
    now = datetime.now(timezone.utc)

    # 1. Create AllegationStatement evidence node
    allegation_node_id = f"allegation-sim-{uuid.uuid4().hex[:8]}"
    append_evidence_node(
        gateway,
        ctx,
        AllegationStatement(
            node_id=allegation_node_id,
            case_id=scenario.case_id,
            source_type=EvidenceSourceType.ALLEGATION,
            created_at=now,
            text=action.claim_text,
            detail_type=AllegationDetailType.UNRECOGNIZED_TRANSACTION,
            classification="dispute_claim",
        ),
    )

    # 2. Create edges linking AllegationStatement -> disputed Transaction(s)
    for i, txn_id in enumerate(action.transaction_node_ids):
        edge_id = f"edge-dispute-{uuid.uuid4().hex[:8]}"
        append_evidence_edge(
            gateway,
            ctx,
            EvidenceEdge(
                edge_id=edge_id,
                case_id=scenario.case_id,
                source_node_id=allegation_node_id,
                target_node_id=txn_id,
                edge_type=EvidenceEdgeType.ALLEGATION,
                created_at=now,
            ),
        )

    # 2b. Mark the transactions as disputed at the evidence node level
    mark_transactions_disputed(gateway, ctx, scenario.case_id, action.transaction_node_ids)

    # 3. Look up transaction details to build the SYSTEM event text
    all_nodes = gateway.evidence_store.get_nodes_by_case(scenario.case_id)
    txn_details = []
    for txn_id in action.transaction_node_ids:
        txn = next((n for n in all_nodes if n.get("node_id") == txn_id), None)
        if txn:
            amount = txn.get("amount", "?")
            merchant = txn.get("merchant_name", "unknown")
            date = txn.get("transaction_date", "unknown")
            auth = txn.get("auth_method", "unknown")
            channel = txn.get("channel", "unknown")
            txn_details.append(f"${amount} {merchant} ({date}, {auth}, {channel})")

    txn_summary = "; ".join(txn_details) if txn_details else "unknown transactions"
    system_text = (
        f"SYSTEM: CCP has marked the following transaction(s) as disputed: "
        f"{txn_summary}. "
        f"Cardmember's claim: {action.claim_text}"
    )

    print(
        f"\n  {DIM}[DisputeAction] Created allegation node {allegation_node_id} "
        f"linked to {len(action.transaction_node_ids)} transaction(s){RESET}"
    )

    # 4. Inject as a SYSTEM event into the copilot
    return await _inject_system_event(
        turn, system_text, scenario, copilot, conversation_history, gateway
    )


# ---------------------------------------------------------------------------
# Phase 2 modes
# ---------------------------------------------------------------------------


async def _phase2_replay(
    transcript_path: str,
    copilot: CopilotOrchestrator,
    scenario: Scenario,
    gateway,
) -> CopilotSuggestion | None:
    """Phase 2 (Mode 1): Replay an existing transcript through the copilot.

    Loads events from a transcript JSON file and feeds them sequentially to the
    copilot orchestrator. No LLM CCP/CM generation — the conversation already
    exists in the transcript.

    Returns the last CopilotSuggestion produced.
    """
    events = load_transcript_file(transcript_path)
    print(f"  Loaded {len(events)} events from {transcript_path}")

    last_suggestion = None
    total = len(events)
    for i, event in enumerate(events, 1):
        _print_turn(i, event.speaker.value, event.text)

        t0 = time.perf_counter()
        suggestion = await copilot.process_event(event, is_last=(i == total))
        copilot_dur = (time.perf_counter() - t0) * 1000

        _persist_trace(
            gateway,
            scenario.case_id,
            "transcript",
            "conversation_turn",
            json.dumps({"turn": i, "speaker": event.speaker.value}),
            json.dumps({"turn": i, "speaker": event.speaker.value, "text": event.text}),
        )

        if suggestion is not None:
            last_suggestion = suggestion
            _print_copilot_brief(suggestion)
            _persist_trace(
                gateway,
                scenario.case_id,
                "copilot_suggestion",
                "suggestion",
                json.dumps({"turn": i}),
                suggestion.model_dump_json(),
                duration_ms=copilot_dur,
            )

    return last_suggestion


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------


async def run_scenario(scenario: Scenario, transcript_path: str | None = None) -> None:
    """Run a full end-to-end simulation for the given scenario."""
    wall_start = time.perf_counter()

    # -- Banner --
    mode_label = "Transcript Replay" if transcript_path else "Live Bedrock LLM"
    print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  AMEX Fraud Servicing — Full E2E Simulation{RESET}")
    print(f"{BOLD}{CYAN}  ({mode_label}){RESET}")
    print(f"{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"\n{BOLD}Scenario: {scenario.title}{RESET}")
    print(f"{DIM}{scenario.description}{RESET}")

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

    # Create simulator provider and CM agent only for generate mode (no transcript)
    simulator_provider = None
    scenario_cm_agent = None
    if not transcript_path:
        # Faster/cheaper Haiku provider for CM/CCP dialogue simulators.
        # Specialist agents (triage, hypothesis, etc.) still use the main provider.
        if settings.llm_provider == "bedrock":
            from copy import copy

            haiku_settings = copy(settings)
            haiku_settings.aws_bedrock_model_id = _HAIKU_MODEL_ID
            simulator_provider = BedrockModelProvider(haiku_settings)
            print(f"  {DIM}Simulator model (CM/CCP): {_HAIKU_MODEL_ID}{RESET}")
        else:
            simulator_provider = model_provider
        scenario_cm_agent = Agent(name="cm_simulator", instructions=scenario.cm_system_prompt)

    db_dir = f"data/simulation/{scenario.name}"
    gateway = create_gateway(db_dir)
    print(f"\n{DIM}Gateway created with SQLite stores in {db_dir}/{RESET}")

    # ===================================================================
    # Phase 1: Seed Evidence
    # ===================================================================
    _print_header("Phase 1: Seeding Evidence Store")

    scenario.seed_evidence_fn(gateway, scenario.case_id)
    case = scenario.create_case_fn(gateway, scenario.case_id, scenario.call_id)

    nodes = gateway.evidence_store.get_nodes_by_case(scenario.case_id)
    edges = gateway.evidence_store.get_edges_by_case(scenario.case_id)
    print(f"  Seeded {len(nodes)} evidence nodes, {len(edges)} edges")
    print(
        f"  Case {scenario.case_id}: status={case.status.value}, "
        f"allegation={case.allegation_type.value}"
    )

    # ===================================================================
    # Phase 2: Copilot
    # ===================================================================
    copilot = CopilotOrchestrator(gateway, model_provider)
    # Set case_id/call_id directly so the retrieval agent queries the
    # correct case — the auto-generated ID from call_id doesn't match
    # the seeded evidence.
    copilot.case_id = scenario.case_id
    copilot.call_id = scenario.call_id

    if transcript_path:
        # -- Mode 1: Replay existing transcript --
        _print_header("Phase 2: Copilot — Transcript Replay")
        print(f"  {DIM}Replaying: {transcript_path}{RESET}")
        last_suggestion = await _phase2_replay(transcript_path, copilot, scenario, gateway)
    else:
        # -- Mode 2: Generate conversation via LLM --
        _print_header("Phase 2: Copilot — Live Call Simulation via Bedrock")
        conversation_history: list[tuple[str, str]] = []
        last_suggestion = None
        turn = 0

        # -- Turn 1: CCP greeting (no copilot context yet) --
        turn += 1
        try:
            t0 = time.perf_counter()
            ccp_text = await generate_ccp_turn(
                conversation_history="[Call begins. Greet the caller.]",
                copilot_context="No copilot data yet — this is the start of the call.",
                model_provider=simulator_provider,
            )
            ccp_dur = (time.perf_counter() - t0) * 1000
        except Exception as exc:
            print(f"\n{RED}CCP generation failed on turn {turn}: {exc}{RESET}")
            sys.exit(1)

        conversation_history.append(("CCP", ccp_text))
        _print_turn(turn, "CCP", ccp_text)

        # CCP turns are NOT processed by the copilot — suggestions are only
        # generated after CARDMEMBER and SYSTEM turns.
        _persist_trace(
            gateway,
            scenario.case_id,
            "transcript",
            "conversation_turn",
            json.dumps({"turn": turn, "speaker": "CCP"}),
            json.dumps({"turn": turn, "speaker": "CCP", "text": ccp_text}),
            duration_ms=ccp_dur,
        )

        # -- Conversation loop --
        while turn < scenario.max_turns:
            # CM turn
            turn += 1
            history_text = "\n".join(f"{spk}: {txt}" for spk, txt in conversation_history)
            try:
                t0 = time.perf_counter()
                cm_text = await generate_cm_turn(
                    scenario_cm_agent, history_text, simulator_provider
                )
                cm_dur = (time.perf_counter() - t0) * 1000
            except Exception as exc:
                print(f"\n{RED}CM generation failed on turn {turn}: {exc}{RESET}")
                break

            conversation_history.append(("CARDMEMBER", cm_text))
            _print_turn(turn, "CARDMEMBER", cm_text)

            t0 = time.perf_counter()
            raw_event = _make_event(scenario.call_id, turn, "CARDMEMBER", cm_text)
            event = parse_transcript_event(raw_event)
            is_last_turn = turn >= scenario.max_turns
            suggestion = await copilot.process_event(event, is_last=is_last_turn)
            copilot_dur = (time.perf_counter() - t0) * 1000

            # Persist CM turn transcript
            _persist_trace(
                gateway,
                scenario.case_id,
                "transcript",
                "conversation_turn",
                json.dumps({"turn": turn, "speaker": "CARDMEMBER"}),
                json.dumps({"turn": turn, "speaker": "CARDMEMBER", "text": cm_text}),
                duration_ms=cm_dur,
            )
            if suggestion is not None:
                last_suggestion = suggestion
                _print_copilot_brief(suggestion)
                _persist_trace(
                    gateway,
                    scenario.case_id,
                    "copilot_suggestion",
                    "suggestion",
                    json.dumps({"turn": turn}),
                    suggestion.model_dump_json(),
                    duration_ms=copilot_dur,
                )

            # Inject SYSTEM events at key points
            if turn == 4 and scenario.system_event_auth:
                # Auth verification event
                turn, _sys_suggestion = await _inject_system_event(
                    turn,
                    scenario.system_event_auth,
                    scenario,
                    copilot,
                    conversation_history,
                    gateway,
                )
                if _sys_suggestion is not None:
                    last_suggestion = _sys_suggestion

                # Evidence event — injected right after auth when early mode is on
                if scenario.inject_evidence_early and scenario.system_event_evidence:
                    turn, _sys_suggestion = await _inject_system_event(
                        turn,
                        scenario.system_event_evidence,
                        scenario,
                        copilot,
                        conversation_history,
                        gateway,
                    )
                    if _sys_suggestion is not None:
                        last_suggestion = _sys_suggestion

            if (
                turn == 10
                and not scenario.inject_evidence_early
                and scenario.system_event_evidence
            ):
                turn, _sys_suggestion = await _inject_system_event(
                    turn,
                    scenario.system_event_evidence,
                    scenario,
                    copilot,
                    conversation_history,
                    gateway,
                )
                if _sys_suggestion is not None:
                    last_suggestion = _sys_suggestion

            # Process dispute actions triggered at this turn
            for action in scenario.dispute_actions:
                if turn == action.trigger_turn:
                    turn, _disp_suggestion = await _process_dispute_action(
                        action, turn, scenario, copilot, conversation_history, gateway
                    )
                    if _disp_suggestion is not None:
                        last_suggestion = _disp_suggestion

            # Check if we have enough turns left for a CCP response
            if turn >= scenario.max_turns:
                break

            # CCP turn (with copilot context)
            turn += 1
            copilot_ctx = _format_copilot_context(last_suggestion)
            history_text = "\n".join(f"{spk}: {txt}" for spk, txt in conversation_history)
            try:
                t0 = time.perf_counter()
                ccp_text = await generate_ccp_turn(history_text, copilot_ctx, simulator_provider)
                ccp_dur = (time.perf_counter() - t0) * 1000
            except Exception as exc:
                print(f"\n{RED}CCP generation failed on turn {turn}: {exc}{RESET}")
                break

            conversation_history.append(("CCP", ccp_text))
            _print_turn(turn, "CCP", ccp_text)

            # CCP turns are NOT processed by the copilot — suggestions are only
            # generated after CARDMEMBER and SYSTEM turns.
            _persist_trace(
                gateway,
                scenario.case_id,
                "transcript",
                "conversation_turn",
                json.dumps({"turn": turn, "speaker": "CCP"}),
                json.dumps({"turn": turn, "speaker": "CCP", "text": ccp_text}),
                duration_ms=ccp_dur,
            )

    # -- Final copilot state --
    specialist_likelihoods = (
        last_suggestion.specialist_likelihoods if last_suggestion else {}
    )
    print(f"\n{BOLD}Final Copilot State:{RESET}")
    print(f"  Hypothesis scores: {json.dumps(copilot.hypothesis_scores, indent=2)}")
    if specialist_likelihoods:
        print(f"  Specialist likelihoods: {json.dumps(specialist_likelihoods, indent=2)}")
    print(f"  Impersonation risk: {copilot.impersonation_risk:.2f}")
    print(f"  Evidence collected: {copilot.evidence_collected}")
    print(f"  Transcript events processed: {len(copilot.transcript_history)}")
    print(f"  Allegations extracted: {len(copilot.accumulated_allegations)}")

    # Persist final copilot state for dashboard
    _persist_trace(
        gateway,
        scenario.case_id,
        "copilot_final",
        "final_state",
        "{}",
        json.dumps(
            {
                "hypothesis_scores": copilot.hypothesis_scores,
                "specialist_likelihoods": specialist_likelihoods,
                "impersonation_risk": copilot.impersonation_risk,
                "evidence_collected": copilot.evidence_collected,
                "allegations_extracted": len(copilot.accumulated_allegations),
            }
        ),
    )

    # ===================================================================
    # Phase 3: Investigator — Post-Call Analysis
    # ===================================================================
    _print_header("Phase 3: Investigator — Post-Call Analysis via Bedrock")

    investigator = InvestigatorOrchestrator(gateway, model_provider)
    try:
        case_pack = await investigator.investigate(scenario.case_id)
    except Exception as exc:
        print(f"\n{RED}Investigation failed: {exc}{RESET}")
        print(f"{DIM}Continuing to verification phase...{RESET}")
        case_pack = None

    if case_pack is not None:
        print(f"\n{BOLD}Case Summary:{RESET}")
        print(f"  {case_pack.case_summary}")

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

        # Persist CasePack for dashboard
        _persist_trace(
            gateway,
            scenario.case_id,
            "investigator",
            "case_pack",
            "{}",
            case_pack.model_dump_json(),
        )

    # ===================================================================
    # Phase 4: Post-Simulation Verification
    # ===================================================================
    _print_header("Phase 4: Post-Simulation Verification")

    final_case = gateway.case_store.get_case(scenario.case_id)
    if final_case:
        print(f"  Case status: {final_case.status.value}")
    else:
        print(f"  {RED}Case not found in store!{RESET}")

    final_nodes = gateway.evidence_store.get_nodes_by_case(scenario.case_id)
    final_edges = gateway.evidence_store.get_edges_by_case(scenario.case_id)
    print(f"  Evidence nodes: {len(final_nodes)} (started with {len(nodes)})")
    print(f"  Evidence edges: {len(final_edges)} (started with {len(edges)})")

    inv_notes = [n for n in final_nodes if n.get("node_type") == "INVESTIGATOR_NOTE"]
    print(f"  InvestigatorNote nodes: {len(inv_notes)}")

    traces = gateway.trace_store.get_traces_by_case(scenario.case_id)
    print(f"  Trace records: {len(traces)}")

    elapsed = time.perf_counter() - wall_start
    minutes, seconds = divmod(elapsed, 60)
    print(f"\n{BOLD}{GREEN}Simulation complete.{RESET}")
    print(f"  Elapsed time: {int(minutes)}m {seconds:.1f}s")
    all_ok = final_case is not None and len(final_nodes) > len(nodes) and len(traces) > 0
    if all_ok:
        print(f"{GREEN}All verification checks passed.{RESET}")
    else:
        print(f"{YELLOW}Some verification checks may have issues — review output above.{RESET}")


def main() -> None:
    """Parse CLI args and run the selected scenario."""
    parser = argparse.ArgumentParser(description="AMEX Fraud Servicing — E2E Simulation Runner")
    parser.add_argument(
        "--scenario",
        "-s",
        default="scam_techvault",
        help=f"Scenario to run. Available: {', '.join(list_scenarios())}",
    )
    parser.add_argument(
        "--transcript",
        "-t",
        default=None,
        help="Path to transcript JSON file. When provided, replays the transcript "
        "instead of generating conversation via LLM.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available scenarios and exit.",
    )
    args = parser.parse_args()

    if args.list:
        print("Available scenarios:")
        for name in list_scenarios():
            s = get_scenario(name)
            print(f"  {name:25s} — {s.title}")
        return

    scenario = get_scenario(args.scenario)

    # Save output to scenario-specific file
    output_dir = f"data/simulation/{scenario.name}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "simulation_output.txt")

    with open(output_path, "w") as log_file:
        original_stdout = sys.stdout
        sys.stdout = TeeWriter(original_stdout, log_file)
        try:
            asyncio.run(run_scenario(scenario, transcript_path=args.transcript))
        finally:
            sys.stdout = original_stdout

    print(f"\nOutput saved to {output_path}")


if __name__ == "__main__":
    main()
