"""Forked simulation runner: replay real transcript, then fork to LLM generation.

Replays the first N events from a real transcript through the copilot to build
up hypothesis scores and probing questions. Then forks: the CCP LLM follows
the copilot's probing questions while the CM LLM continues based on the
established conversation and user-provided instructions.

This tests "what would happen if the CCP asked our probing questions?" against
a real call's conversational context.

Usage:
    python scripts/run_forked_simulation.py \
        --transcript data/transcripts/real_call.json \
        --fork-after 10 \
        --cm-instructions "You went to the store but claim you didn't."

    # With evidence seeding from an existing scenario:
    python scripts/run_forked_simulation.py \
        --transcript data/transcripts/real_call.json \
        --fork-after 10 \
        --cm-instructions "..." \
        --scenario scam_techvault

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
from copy import copy
from datetime import datetime, timezone

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")

from scripts.simulation_data import discover_scenarios  # noqa: E402

discover_scenarios()

from agentic_fraud_servicing.config import get_settings  # noqa: E402
from agentic_fraud_servicing.copilot.orchestrator import CopilotOrchestrator  # noqa: E402
from agentic_fraud_servicing.gateway.tool_gateway import AuthContext  # noqa: E402
from agentic_fraud_servicing.gateway.tools.write_tools import create_case  # noqa: E402
from agentic_fraud_servicing.ingestion.redaction import redact_all  # noqa: E402
from agentic_fraud_servicing.ingestion.transcript import parse_transcript_event  # noqa: E402
from agentic_fraud_servicing.models.case import AuditEntry, Case  # noqa: E402
from agentic_fraud_servicing.models.enums import CaseStatus  # noqa: E402
from agentic_fraud_servicing.providers.base import get_model_provider  # noqa: E402
from agentic_fraud_servicing.providers.bedrock_provider import BedrockModelProvider  # noqa: E402
from agentic_fraud_servicing.ui.helpers import create_gateway  # noqa: E402
from scripts.simulation_data import (  # noqa: E402
    create_fork_cm_agent,
    generate_ccp_turn_forked,
    generate_cm_turn,
    get_scenario,
)

# ---------------------------------------------------------------------------
# ANSI color codes
# ---------------------------------------------------------------------------

BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
DIM = "\033[2m"
RESET = "\033[0m"

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")

# Haiku model for CM/CCP simulators
_HAIKU_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


class TeeWriter(io.TextIOBase):
    """Duplicates writes to both the terminal and a plain-text log file."""

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


def _print_turn(turn: int, speaker: str, text: str, forked: bool = False) -> None:
    """Print a conversation turn with color formatting."""
    color = CYAN if speaker == "CCP" else YELLOW if speaker == "CARDMEMBER" else DIM
    tag = f" {GREEN}[FORKED]{RESET}" if forked else ""
    print(f"\n{BOLD}[Turn {turn}]{RESET} {color}{speaker}:{RESET}{tag} {text}")


def _print_copilot_brief(suggestion) -> None:
    """Print copilot suggestion summary."""
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
    if suggestion.probing_questions:
        print(f"  {CYAN}Probing Questions:{RESET}")
        for pq in suggestion.probing_questions:
            status = pq.get("status", "?")
            if status == "answered":
                color = GREEN
            elif status == "invalidated":
                color = RED
            elif status == "skipped":
                color = YELLOW
            else:
                color = CYAN
            target = pq.get("target_category", "")
            target_str = f" [{target}]" if target else ""
            print(f"    {color}[{status}]{RESET}{target_str} {pq['text']}")
    elif suggestion.suggested_questions:
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


def _persist_trace(
    gateway,
    case_id: str,
    agent_id: str,
    action: str,
    input_data: str,
    output_data: str,
    duration_ms: float = 0.0,
) -> None:
    """Persist a data record to the trace store for dashboard consumption."""
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


def _extract_pending_questions(suggestion) -> list[str]:
    """Extract pending probing question texts from a CopilotSuggestion."""
    if suggestion is None:
        return []
    if suggestion.probing_questions:
        return [pq["text"] for pq in suggestion.probing_questions if pq.get("status") == "pending"]
    return suggestion.suggested_questions[:3]


def _print_header(text: str) -> None:
    """Print a section header."""
    print(f"\n{BOLD}{GREEN}{'=' * 60}{RESET}")
    print(f"{BOLD}{GREEN}{text}{RESET}")
    print(f"{BOLD}{GREEN}{'=' * 60}{RESET}")


def _load_transcript(path: str) -> list[dict]:
    """Load raw transcript events from a JSON file.

    Returns the raw dicts (not parsed TranscriptEvents) so we can access
    the original speaker strings and feed them through parse_transcript_event
    during replay.
    """
    with open(path, encoding="utf-8") as f:
        events = json.load(f)
    if not isinstance(events, list):
        raise ValueError(f"Expected JSON array, got {type(events).__name__}")
    return events


# ---------------------------------------------------------------------------
# Main forked simulation
# ---------------------------------------------------------------------------


async def run_forked_simulation(
    transcript_path: str,
    fork_after: int,
    cm_instructions: str,
    scenario_name: str | None = None,
    min_forked_turns: int = 10,
    sim_name: str | None = None,
) -> None:
    """Run a forked simulation: replay real transcript, then fork to LLM.

    Args:
        transcript_path: Path to the real transcript JSON file.
        fork_after: Number of events to replay before forking.
        cm_instructions: User-provided instructions for the CM agent.
        scenario_name: Optional scenario name for evidence seeding.
        min_forked_turns: Minimum turns to generate before allowing early stop.
        sim_name: Output directory name. Defaults to {scenario}_fork_turn{N}.
    """
    wall_start = time.perf_counter()

    # -- Load transcript --
    raw_events = _load_transcript(transcript_path)
    if fork_after > len(raw_events):
        print(
            f"{RED}--fork-after ({fork_after}) exceeds transcript length "
            f"({len(raw_events)} events). Using all events.{RESET}"
        )
        fork_after = len(raw_events)

    replay_events = raw_events[:fork_after]
    call_id = replay_events[0]["call_id"] if replay_events else f"call-fork-{uuid.uuid4().hex[:8]}"
    case_id = f"case-fork-{uuid.uuid4().hex[:8]}"

    # -- Banner --
    print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  AMEX Fraud Servicing — Forked Simulation{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"\n{DIM}Transcript: {transcript_path}{RESET}")
    print(
        f"{DIM}Total events: {len(raw_events)} | Replaying: {fork_after}"
        f" | Then forking to LLM{RESET}"
    )
    if scenario_name:
        print(f"{DIM}Evidence from scenario: {scenario_name}{RESET}")
    truncated = cm_instructions[:100] + ("..." if len(cm_instructions) > 100 else "")
    print(f"{DIM}CM instructions: {truncated}{RESET}")

    # -- Setup providers --
    try:
        settings = get_settings()
    except Exception as exc:
        print(f"\n{RED}Error loading settings: {exc}{RESET}")
        sys.exit(1)

    model_provider = get_model_provider(settings)

    if settings.llm_provider == "bedrock":
        haiku_settings = copy(settings)
        haiku_settings.aws_bedrock_model_id = _HAIKU_MODEL_ID
        simulator_provider = BedrockModelProvider(haiku_settings)
        print(f"  {DIM}Simulator model (CM/CCP): {_HAIKU_MODEL_ID}{RESET}")
    else:
        simulator_provider = model_provider

    # -- Setup gateway and optional evidence seeding --
    if not sim_name:
        base = scenario_name or os.path.splitext(os.path.basename(transcript_path))[0]
        sim_name = f"{base}_fork_turn{fork_after}"
    db_dir = f"data/simulation/{sim_name}"
    gateway = create_gateway(db_dir)
    print(f"{DIM}Gateway created with SQLite stores in {db_dir}/{RESET}")

    if scenario_name:
        scenario = get_scenario(scenario_name)
        # Override IDs to match our forked simulation
        scenario.case_id = case_id
        scenario.call_id = call_id
        scenario.seed_evidence_fn(gateway, case_id)
        scenario.create_case_fn(gateway, case_id, call_id)
        nodes = gateway.evidence_store.get_nodes_by_case(case_id)
        edges = gateway.evidence_store.get_edges_by_case(case_id)
        print(f"  Seeded {len(nodes)} evidence nodes, {len(edges)} edges from {scenario_name}")
    else:
        # Create a minimal case so the copilot and trace store have a valid case_id
        ctx = AuthContext(agent_id="simulation", case_id=case_id, permissions={"write"})
        now = datetime.now(timezone.utc)
        minimal_case = Case(
            case_id=case_id,
            call_id=call_id,
            customer_id="unknown",
            account_id="unknown",
            status=CaseStatus.OPEN,
            audit_trail=[
                AuditEntry(
                    timestamp=now,
                    action="case_created",
                    agent_id="simulation",
                    details="Minimal case for forked simulation (no scenario evidence).",
                ),
            ],
            created_at=now,
        )
        create_case(gateway, ctx, minimal_case)
        print(f"  {DIM}Created minimal case (no evidence seeded){RESET}")

    # -- Setup copilot --
    copilot = CopilotOrchestrator(gateway, model_provider)
    copilot.case_id = case_id
    copilot.call_id = call_id

    # ===================================================================
    # Phase A: Replay real transcript events
    # ===================================================================
    _print_header(f"Phase A: Replaying {fork_after} Events from Real Transcript")

    conversation_history: list[tuple[str, str]] = []
    last_suggestion = None

    for i, raw_event in enumerate(replay_events, 1):
        speaker = raw_event["speaker"]
        text = raw_event["text"]
        conversation_history.append((speaker, text))
        _print_turn(i, speaker, text)

        # Feed to copilot (only CARDMEMBER and SYSTEM events produce suggestions)
        event = parse_transcript_event(raw_event)
        t0 = time.perf_counter()
        suggestion = await copilot.process_event(event, is_last=False)
        copilot_dur = (time.perf_counter() - t0) * 1000

        _persist_trace(
            gateway,
            case_id,
            "transcript",
            "conversation_turn",
            json.dumps({"turn": i, "speaker": speaker, "phase": "replay"}),
            json.dumps({"turn": i, "speaker": speaker, "text": text}),
        )

        if suggestion is not None:
            last_suggestion = suggestion
            _print_copilot_brief(suggestion)
            _persist_trace(
                gateway,
                case_id,
                "copilot_suggestion",
                "suggestion",
                json.dumps({"turn": i, "phase": "replay"}),
                suggestion.model_dump_json(),
                duration_ms=copilot_dur,
            )

    # -- Show copilot state at fork point --
    pending_qs = _extract_pending_questions(last_suggestion)
    print(f"\n{BOLD}--- Fork Point (after event {fork_after}) ---{RESET}")
    print(f"  Hypothesis: {json.dumps(copilot.hypothesis_scores, indent=2)}")
    if pending_qs:
        print(f"  {CYAN}Pending probing questions for CCP:{RESET}")
        for q in pending_qs:
            print(f"    - {q}")
    else:
        print(f"  {DIM}No pending probing questions at fork point{RESET}")

    # ===================================================================
    # Phase B: Forked LLM generation
    # ===================================================================
    _print_header("Phase B: Forked Generation — CCP Follows Probing Questions")

    cm_agent = create_fork_cm_agent(cm_instructions)
    print(f"\n  {DIM}CM instructions: {cm_instructions}{RESET}")
    turn = fork_after

    # CCP always goes first after the fork — the whole point is to test
    # whether asking the copilot's probing questions surfaces useful info.
    # Run at least min_forked_turns, then stop when information_sufficient.
    forked_turns = 0
    while True:
        # -- CCP turn (guided by probing questions) --
        turn += 1
        forked_turns += 1
        pending_qs = _extract_pending_questions(last_suggestion)
        history_text = "\n".join(f"{spk}: {txt}" for spk, txt in conversation_history)

        try:
            t0 = time.perf_counter()
            ccp_text = await generate_ccp_turn_forked(history_text, pending_qs, simulator_provider)
            ccp_dur = (time.perf_counter() - t0) * 1000
        except Exception as exc:
            print(f"\n{RED}CCP generation failed on turn {turn}: {exc}{RESET}")
            break

        conversation_history.append(("CCP", ccp_text))
        _print_turn(turn, "CCP", ccp_text, forked=True)

        if pending_qs:
            print(f"  {DIM}(Suggested questions: {'; '.join(pending_qs[:3])}){RESET}")

        _persist_trace(
            gateway,
            case_id,
            "transcript",
            "conversation_turn",
            json.dumps({"turn": turn, "speaker": "CCP", "phase": "forked"}),
            json.dumps({"turn": turn, "speaker": "CCP", "text": ccp_text}),
            duration_ms=ccp_dur,
        )

        # (continue to CM turn)

        # -- CM turn --
        turn += 1
        forked_turns += 1
        history_text = "\n".join(f"{spk}: {txt}" for spk, txt in conversation_history)

        try:
            t0 = time.perf_counter()
            cm_text = await generate_cm_turn(cm_agent, history_text, simulator_provider)
            cm_dur = (time.perf_counter() - t0) * 1000
        except Exception as exc:
            print(f"\n{RED}CM generation failed on turn {turn}: {exc}{RESET}")
            break

        conversation_history.append(("CARDMEMBER", cm_text))
        _print_turn(turn, "CARDMEMBER", cm_text, forked=True)

        # Feed CM turn to copilot
        raw_event = _make_event(call_id, turn, "CARDMEMBER", cm_text)
        event = parse_transcript_event(raw_event)
        is_last_cm = False  # we don't know when the conversation ends
        t0 = time.perf_counter()
        suggestion = await copilot.process_event(event, is_last=is_last_cm)
        copilot_dur = (time.perf_counter() - t0) * 1000

        _persist_trace(
            gateway,
            case_id,
            "transcript",
            "conversation_turn",
            json.dumps({"turn": turn, "speaker": "CARDMEMBER", "phase": "forked"}),
            json.dumps({"turn": turn, "speaker": "CARDMEMBER", "text": cm_text}),
            duration_ms=cm_dur,
        )

        if suggestion is not None:
            last_suggestion = suggestion
            _print_copilot_brief(suggestion)
            _persist_trace(
                gateway,
                case_id,
                "copilot_suggestion",
                "suggestion",
                json.dumps({"turn": turn, "phase": "forked"}),
                suggestion.model_dump_json(),
                duration_ms=copilot_dur,
            )

        # Stop once minimum turns reached and copilot has enough information
        if forked_turns >= min_forked_turns and suggestion and suggestion.information_sufficient:
            print(f"\n  {GREEN}Copilot: information sufficient — ending conversation{RESET}")
            break

    # ===================================================================
    # Final state
    # ===================================================================
    _print_header("Final Copilot State")

    specialist_likelihoods = last_suggestion.specialist_likelihoods if last_suggestion else {}
    print(f"  Hypothesis scores: {json.dumps(copilot.hypothesis_scores, indent=2)}")
    if specialist_likelihoods:
        print(f"  Specialist likelihoods: {json.dumps(specialist_likelihoods, indent=2)}")
    print(f"  Impersonation risk: {copilot.impersonation_risk:.2f}")
    print(f"  Evidence collected: {copilot.evidence_collected}")
    print(f"  Transcript events processed: {len(copilot.transcript_history)}")
    print(f"  Allegations extracted: {len(copilot.accumulated_allegations)}")
    print(f"  Replay turns: {fork_after} | Forked turns: {forked_turns}")

    # Persist final state
    _persist_trace(
        gateway,
        case_id,
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
                "fork_after": fork_after,
                "forked_turns": forked_turns,
            }
        ),
    )

    elapsed = time.perf_counter() - wall_start
    minutes, seconds = divmod(elapsed, 60)
    print(f"\n{BOLD}{GREEN}Forked simulation complete.{RESET}")
    print(f"  Elapsed time: {int(minutes)}m {seconds:.1f}s")
    print(f"  Results in: {db_dir}/")


def main() -> None:
    """Parse CLI args and run the forked simulation."""
    parser = argparse.ArgumentParser(description="AMEX Fraud Servicing — Forked Simulation Runner")
    parser.add_argument(
        "--transcript",
        "-t",
        required=True,
        help="Path to real transcript JSON file.",
    )
    parser.add_argument(
        "--fork-after",
        "-f",
        type=int,
        required=True,
        help="Number of transcript events to replay before forking to LLM.",
    )
    parser.add_argument(
        "--cm-instructions",
        "-c",
        required=True,
        help="Instructions for the CM agent (backstory, what they know/hide).",
    )
    parser.add_argument(
        "--scenario",
        "-s",
        default=None,
        help="Optional scenario name for evidence seeding (e.g., scam_techvault).",
    )
    parser.add_argument(
        "--min-forked-turns",
        "-m",
        type=int,
        default=10,
        help="Minimum forked turns before allowing early stop (default: 10).",
    )
    parser.add_argument(
        "--name",
        "-n",
        default=None,
        help="Output directory name under data/simulation/. "
        "Defaults to {scenario}_fork_turn{N} or {transcript}_fork_turn{N}.",
    )
    args = parser.parse_args()

    # Validate transcript exists
    if not os.path.isfile(args.transcript):
        print(f"{RED}Transcript file not found: {args.transcript}{RESET}")
        sys.exit(1)

    # Determine output directory and set up logging
    if args.name:
        sim_name = args.name
    else:
        base = args.scenario or os.path.splitext(os.path.basename(args.transcript))[0]
        sim_name = f"{base}_fork_turn{args.fork_after}"
    output_dir = f"data/simulation/{sim_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "simulation_output.txt")

    with open(output_path, "w") as log_file:
        original_stdout = sys.stdout
        sys.stdout = TeeWriter(original_stdout, log_file)
        try:
            asyncio.run(
                run_forked_simulation(
                    transcript_path=args.transcript,
                    fork_after=args.fork_after,
                    cm_instructions=args.cm_instructions,
                    scenario_name=args.scenario,
                    min_forked_turns=args.min_forked_turns,
                    sim_name=sim_name,
                )
            )
        finally:
            sys.stdout = original_stdout

    print(f"\nOutput saved to {output_path}")


if __name__ == "__main__":
    main()
