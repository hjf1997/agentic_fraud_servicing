"""Enterprise evaluation runner: replay transcripts through CopilotOrchestrator.

Replays pre-existing transcript JSON files through the copilot pipeline with
real Bedrock LLM, capturing per-turn latency and structured evaluation data.
No investigator phase — copilot-only evaluation.

Usage:
    python scripts/run_evaluation.py --scenario scam_techvault
    python scripts/run_evaluation.py --scenario scam_techvault --transcript path/to/transcript.json
    python scripts/run_evaluation.py --list

Requires valid AWS credentials in .env (LLM_PROVIDER=bedrock).
"""

import argparse
import asyncio
import importlib
import io
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone

# Ensure project root is on sys.path so 'scripts' package is importable
# when running directly via: python scripts/run_evaluation.py
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Suppress "OPENAI_API_KEY is not set, skipping trace export" noise from the
# Agents SDK when using Bedrock provider instead of OpenAI.
os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")

# Import scenario modules to trigger registration
import scripts.scenario_dispute_to_fraud  # noqa: E402, F401
import scripts.scenario_doordash_dashpass  # noqa: E402, F401
import scripts.scenario_doordash_dashpass_v2  # noqa: E402, F401
import scripts.scenario_doordash_fraud  # noqa: E402, F401
import scripts.scenario_highrisk_merchant  # noqa: E402, F401
import scripts.scenario_scam_techvault  # noqa: E402, F401
from agentic_fraud_servicing.config import get_settings  # noqa: E402
from agentic_fraud_servicing.copilot.orchestrator import CopilotOrchestrator  # noqa: E402
from agentic_fraud_servicing.evaluation.models import EvaluationRun, TurnMetric  # noqa: E402
from agentic_fraud_servicing.ingestion.redaction import redact_all  # noqa: E402
from agentic_fraud_servicing.ingestion.transcript import parse_transcript_json  # noqa: E402
from agentic_fraud_servicing.providers.base import get_model_provider  # noqa: E402
from agentic_fraud_servicing.ui.helpers import create_gateway  # noqa: E402
from scripts.simulation_data import get_scenario, list_scenarios  # noqa: E402

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


def _load_ground_truth(scenario_name: str) -> dict:
    """Load GROUND_TRUTH from the scenario module, or return empty dict."""
    module_name = f"scripts.scenario_{scenario_name}"
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, "GROUND_TRUTH", {})
    except (ImportError, AttributeError):
        return {}


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


def _top_hypothesis(scores: dict[str, float]) -> str:
    """Return the top-scoring hypothesis category and its score."""
    if not scores:
        return "N/A"
    top_key = max(scores, key=scores.get)
    return f"{top_key}={scores[top_key]:.2f}"


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


async def run_evaluation(scenario_name: str, transcript_path: str) -> None:
    """Replay a transcript through the copilot and capture evaluation metrics."""
    wall_start = time.perf_counter()
    start_time = datetime.now(timezone.utc).isoformat()

    scenario = get_scenario(scenario_name)
    ground_truth = _load_ground_truth(scenario_name)

    # -- Banner --
    print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  AMEX Fraud Servicing — Evaluation Runner{RESET}")
    print(f"{BOLD}{CYAN}  (Copilot Replay Mode){RESET}")
    print(f"{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"\n{BOLD}Scenario: {scenario.title}{RESET}")
    print(f"{DIM}{scenario.description}{RESET}")
    print(f"{DIM}Transcript: {transcript_path}{RESET}")
    if ground_truth:
        gt_cat = ground_truth.get("investigation_category", "?")
        print(f"{DIM}Ground truth category: {gt_cat}{RESET}")

    # -- Setup --
    try:
        settings = get_settings()
    except Exception as exc:
        print(f"\n{RED}Error loading settings: {exc}{RESET}")
        print(f"{RED}Ensure .env is configured with LLM_PROVIDER, AWS_PROFILE, etc.{RESET}")
        sys.exit(1)

    try:
        model_provider = get_model_provider(settings)
    except Exception as exc:
        print(f"\n{RED}Error creating model provider: {exc}{RESET}")
        print(f"{RED}Check AWS credentials and Bedrock model configuration.{RESET}")
        sys.exit(1)

    db_dir = f"data/evaluations/{scenario_name}"
    gateway = create_gateway(db_dir)
    print(f"{DIM}Gateway created with SQLite stores in {db_dir}/{RESET}")

    # ===================================================================
    # Phase 1: Seed Evidence
    # ===================================================================
    print(f"\n{BOLD}{GREEN}{'=' * 60}{RESET}")
    print(f"{BOLD}{GREEN}Phase 1: Seeding Evidence Store{RESET}")
    print(f"{BOLD}{GREEN}{'=' * 60}{RESET}")

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
    # Phase 2: Copilot Replay
    # ===================================================================
    print(f"\n{BOLD}{GREEN}{'=' * 60}{RESET}")
    print(f"{BOLD}{GREEN}Phase 2: Copilot — Transcript Replay{RESET}")
    print(f"{BOLD}{GREEN}{'=' * 60}{RESET}")

    # Load transcript
    with open(transcript_path) as f:
        transcript_json = f.read()
    events = parse_transcript_json(transcript_json)
    print(f"  Loaded {len(events)} transcript events")

    copilot = CopilotOrchestrator(gateway, model_provider)
    copilot.case_id = scenario.case_id
    copilot.call_id = scenario.call_id

    turn_metrics: list[TurnMetric] = []
    total_latency_ms = 0.0
    prev_allegation_count = 0

    for i, event in enumerate(events, 1):
        t0 = time.perf_counter()
        suggestion = await copilot.process_event(event)
        latency_ms = (time.perf_counter() - t0) * 1000
        total_latency_ms += latency_ms

        # Capture only new allegations from this turn
        current_count = len(copilot.accumulated_allegations)
        new_allegations = copilot.accumulated_allegations[prev_allegation_count:]
        prev_allegation_count = current_count

        metric = TurnMetric(
            turn_number=i,
            speaker=event.speaker.value,
            text=event.text,
            latency_ms=latency_ms,
            copilot_suggestion=suggestion.model_dump(mode="json"),
            hypothesis_scores=dict(copilot.hypothesis_scores),
            allegations_extracted=[a.model_dump(mode="json") for a in new_allegations],
        )
        turn_metrics.append(metric)

        # Print progress
        top = _top_hypothesis(copilot.hypothesis_scores)
        color = CYAN if event.speaker.value == "CCP" else YELLOW
        if event.speaker.value == "SYSTEM":
            color = DIM
        print(
            f"  {BOLD}[Turn {i}/{len(events)}]{RESET} "
            f"{color}{event.speaker.value}{RESET} "
            f"| {latency_ms:7.0f}ms "
            f"| top: {top}"
        )

        # Persist transcript turn and copilot suggestion
        _persist_trace(
            gateway,
            scenario.case_id,
            "transcript",
            "conversation_turn",
            json.dumps({"turn": i, "speaker": event.speaker.value}),
            json.dumps({"turn": i, "speaker": event.speaker.value, "text": event.text}),
            duration_ms=latency_ms,
        )
        _persist_trace(
            gateway,
            scenario.case_id,
            "copilot_suggestion",
            "suggestion",
            json.dumps({"turn": i}),
            suggestion.model_dump_json(),
            duration_ms=latency_ms,
        )

    # -- Final copilot state --
    copilot_final_state = {
        "hypothesis_scores": copilot.hypothesis_scores,
        "impersonation_risk": copilot.impersonation_risk,
        "missing_fields": copilot.missing_fields,
        "evidence_collected": copilot.evidence_collected,
        "allegations_extracted": len(copilot.accumulated_allegations),
    }

    _persist_trace(
        gateway,
        scenario.case_id,
        "copilot_final",
        "final_state",
        "{}",
        json.dumps(copilot_final_state),
    )

    # -- Build EvaluationRun --
    end_time = datetime.now(timezone.utc).isoformat()
    evaluation_run = EvaluationRun(
        scenario_name=scenario_name,
        ground_truth=ground_truth,
        turn_metrics=turn_metrics,
        total_turns=len(turn_metrics),
        total_latency_ms=total_latency_ms,
        start_time=start_time,
        end_time=end_time,
        copilot_final_state=copilot_final_state,
    )

    # Save EvaluationRun as JSON
    output_dir = f"data/evaluations/{scenario_name}"
    os.makedirs(output_dir, exist_ok=True)
    eval_run_path = os.path.join(output_dir, "evaluation_run.json")
    with open(eval_run_path, "w") as f:
        f.write(evaluation_run.model_dump_json(indent=2))

    # -- Summary --
    print(f"\n{BOLD}{GREEN}{'=' * 60}{RESET}")
    print(f"{BOLD}{GREEN}Evaluation Summary{RESET}")
    print(f"{BOLD}{GREEN}{'=' * 60}{RESET}")
    print(f"  Total turns: {len(turn_metrics)}")
    print(f"  Total latency: {total_latency_ms:.0f}ms")
    print(f"  Avg latency per turn: {total_latency_ms / max(len(turn_metrics), 1):.0f}ms")
    print(f"  Final hypothesis scores: {json.dumps(copilot.hypothesis_scores, indent=2)}")
    print(f"  Allegations extracted: {len(copilot.accumulated_allegations)}")
    if ground_truth:
        print(f"  Ground truth: {json.dumps(ground_truth, indent=2)}")
    else:
        print(f"  {YELLOW}No ground truth defined for this scenario{RESET}")

    elapsed = time.perf_counter() - wall_start
    minutes, seconds = divmod(elapsed, 60)
    print(f"\n{BOLD}{GREEN}Evaluation complete.{RESET}")
    print(f"  Elapsed time: {int(minutes)}m {seconds:.1f}s")
    print(f"  Results saved to: {eval_run_path}")


def main() -> None:
    """Parse CLI args and run the evaluation."""
    parser = argparse.ArgumentParser(
        description="AMEX Fraud Servicing — Evaluation Runner (Copilot Replay)"
    )
    parser.add_argument(
        "--scenario",
        "-s",
        help=f"Scenario to evaluate. Available: {', '.join(list_scenarios())}",
    )
    parser.add_argument(
        "--transcript",
        "-t",
        help="Override transcript file path (default: scripts/transcripts/{scenario}.json)",
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
            transcript = os.path.join("scripts", "transcripts", f"{name}.json")
            exists = "yes" if os.path.isfile(transcript) else "no"
            print(f"  {name:30s} — {s.title}  (transcript: {exists})")
        return

    if not args.scenario:
        parser.error("--scenario is required (or use --list to see available scenarios)")

    scenario_name = args.scenario
    # Validate scenario exists
    get_scenario(scenario_name)

    # Determine transcript path
    if args.transcript:
        transcript_path = args.transcript
    else:
        transcript_path = os.path.join("scripts", "transcripts", f"{scenario_name}.json")

    if not os.path.isfile(transcript_path):
        print(f"{RED}Error: Transcript file not found: {transcript_path}{RESET}")
        print(
            f"{YELLOW}Create a transcript JSON file or use --transcript to specify a path.{RESET}"
        )
        sys.exit(1)

    # Save output to scenario-specific file
    output_dir = f"data/evaluations/{scenario_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "evaluation_output.txt")

    with open(output_path, "w") as log_file:
        original_stdout = sys.stdout
        sys.stdout = TeeWriter(original_stdout, log_file)
        try:
            asyncio.run(run_evaluation(scenario_name, transcript_path))
        finally:
            sys.stdout = original_stdout

    print(f"\nOutput saved to {output_path}")


if __name__ == "__main__":
    main()
