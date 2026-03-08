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
import uuid

# Ensure project root is on sys.path so 'scripts' package is importable
# when running directly via: python scripts/run_simulation.py
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Suppress "OPENAI_API_KEY is not set, skipping trace export" noise from the
# Agents SDK when using Bedrock provider instead of OpenAI.
os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")

# Import scenario modules to trigger registration
from agents import Agent  # noqa: E402

import scripts.scenario_doordash_dashpass  # noqa: E402, F401
import scripts.scenario_doordash_fraud  # noqa: E402, F401
import scripts.scenario_highrisk_merchant  # noqa: E402, F401
import scripts.scenario_scam_techvault  # noqa: E402, F401
from agentic_fraud_servicing.config import get_settings  # noqa: E402
from agentic_fraud_servicing.copilot.orchestrator import CopilotOrchestrator  # noqa: E402
from agentic_fraud_servicing.ingestion.transcript import parse_transcript_event  # noqa: E402
from agentic_fraud_servicing.investigator.orchestrator import (
    InvestigatorOrchestrator,  # noqa: E402
)
from agentic_fraud_servicing.models.case import CopilotSuggestion  # noqa: E402
from agentic_fraud_servicing.providers.base import get_model_provider  # noqa: E402
from agentic_fraud_servicing.ui.helpers import create_gateway  # noqa: E402
from scripts.simulation_data import (  # noqa: E402
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
    """Print full copilot suggestions after each turn."""
    scores = suggestion.hypothesis_scores
    print(
        f"  {DIM}Copilot Scores: "
        f"FRAUD={scores.get('FRAUD', 0):.2f} | "
        f"DISPUTE={scores.get('DISPUTE', 0):.2f} | "
        f"SCAM={scores.get('SCAM', 0):.2f} | "
        f"Impersonation={suggestion.impersonation_risk:.2f}{RESET}"
    )
    if suggestion.suggested_questions:
        print(f"  {CYAN}Suggested Questions:{RESET}")
        for q in suggestion.suggested_questions[:3]:
            print(f"    - {q}")
    if suggestion.risk_flags:
        print(f"  {RED}Risk Flags:{RESET}")
        for flag in suggestion.risk_flags[:5]:
            print(f"    ! {flag}")
    if suggestion.running_summary:
        print(f"  {DIM}Summary: {suggestion.running_summary}{RESET}")
    if suggestion.safety_guidance:
        print(f"  {YELLOW}Safety: {suggestion.safety_guidance}{RESET}")


def _print_header(text: str) -> None:
    """Print a section header."""
    print(f"\n{BOLD}{GREEN}{'=' * 60}{RESET}")
    print(f"{BOLD}{GREEN}{text}{RESET}")
    print(f"{BOLD}{GREEN}{'=' * 60}{RESET}")


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------


async def run_scenario(scenario: Scenario) -> None:
    """Run a full end-to-end simulation for the given scenario."""

    # -- Banner --
    print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  AMEX Fraud Servicing — Full E2E Simulation{RESET}")
    print(f"{BOLD}{CYAN}  (Live Bedrock LLM){RESET}")
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

    db_dir = f"data/simulation/{scenario.name}"
    gateway = create_gateway(db_dir)
    print(f"\n{DIM}Gateway created with SQLite stores in {db_dir}/{RESET}")

    # Create CM agent for this scenario
    scenario_cm_agent = Agent(name="cm_simulator", instructions=scenario.cm_system_prompt)

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

    raw_event = _make_event(scenario.call_id, turn, "CCP", ccp_text)
    event = parse_transcript_event(raw_event)
    last_suggestion = await copilot.process_event(event)

    # -- Conversation loop --
    while turn < scenario.max_turns:
        # CM turn
        turn += 1
        history_text = "\n".join(f"{spk}: {txt}" for spk, txt in conversation_history)
        try:
            cm_text = await generate_cm_turn(scenario_cm_agent, history_text, model_provider)
        except Exception as exc:
            print(f"\n{RED}CM generation failed on turn {turn}: {exc}{RESET}")
            break

        conversation_history.append(("CARDMEMBER", cm_text))
        _print_turn(turn, "CARDMEMBER", cm_text)

        raw_event = _make_event(scenario.call_id, turn, "CARDMEMBER", cm_text)
        event = parse_transcript_event(raw_event)
        last_suggestion = await copilot.process_event(event)
        _print_copilot_brief(last_suggestion)

        # Inject SYSTEM events at key points
        if turn == 4:
            turn += 1
            conversation_history.append(("SYSTEM", scenario.system_event_auth))
            _print_turn(turn, "SYSTEM", scenario.system_event_auth)
            raw_event = _make_event(scenario.call_id, turn, "SYSTEM", scenario.system_event_auth)
            event = parse_transcript_event(raw_event)
            last_suggestion = await copilot.process_event(event)
            _print_copilot_brief(last_suggestion)

        if turn == 10:
            turn += 1
            conversation_history.append(("SYSTEM", scenario.system_event_evidence))
            _print_turn(turn, "SYSTEM", scenario.system_event_evidence)
            raw_event = _make_event(
                scenario.call_id, turn, "SYSTEM", scenario.system_event_evidence
            )
            event = parse_transcript_event(raw_event)
            last_suggestion = await copilot.process_event(event)
            _print_copilot_brief(last_suggestion)

        # Check if we have enough turns left for a CCP response
        if turn >= scenario.max_turns:
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

        raw_event = _make_event(scenario.call_id, turn, "CCP", ccp_text)
        event = parse_transcript_event(raw_event)
        last_suggestion = await copilot.process_event(event)

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

    print(f"\n{BOLD}{GREEN}Simulation complete.{RESET}")
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
            asyncio.run(run_scenario(scenario))
        finally:
            sys.stdout = original_stdout

    print(f"\nOutput saved to {output_path}")


if __name__ == "__main__":
    main()
