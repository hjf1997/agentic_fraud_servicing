"""Enterprise evaluation runner: evaluate simulation results from DB.

Loads copilot assessment data from the simulation's SQLite stores (traces.db,
evidence.db) and transcript file, then runs the 8-dimension quality evaluators.
No copilot replay — evaluation reads the simulation's actual results.

Usage:
    python scripts/run_evaluation.py --scenario scam_techvault
    python scripts/run_evaluation.py --scenario scam_techvault --data-dir data/simulation/scam_techvault
    python scripts/run_evaluation.py --list

Requires ConnectChain configuration in .env (LLM_PROVIDER=connectchain) for
LLM-powered evaluators.
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
from datetime import datetime, timezone

# Ensure project root is on sys.path so 'scripts' package is importable
# when running directly via: python scripts/run_evaluation.py
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Suppress "OPENAI_API_KEY is not set, skipping trace export" noise from the
# Agents SDK when not using direct OpenAI.
os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")

# Configure logging so that:
# 1. OPENAI_LOG=debug/info HTTP request logs go to stdout (visible in TeeWriter)
# 2. Agents SDK "Error getting response" messages include HTTP status codes
import logging

_log_fmt = logging.Formatter(
    "[%(asctime)s %(name)s %(levelname)s] %(message)s", datefmt="%H:%M:%S"
)
_stdout_handler = logging.StreamHandler(sys.stdout)
_stdout_handler.setFormatter(_log_fmt)


class _HttpErrorFilter(logging.Filter):
    """Intercept Agents SDK error logs and append HTTP status code details."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.exc_info and record.exc_info[1] is not None:
            exc = record.exc_info[1]
            current = exc
            while current is not None:
                if hasattr(current, "status_code"):
                    code = getattr(current, "status_code", None)
                    body = (
                        getattr(current, "body", None) or getattr(current, "message", None) or ""
                    )
                    record.msg = f"HTTP {code} error — {body}\n(original: {record.msg})"
                    break
                current = getattr(current, "__cause__", None)
        elif hasattr(record, "msg") and "Error getting response" in str(record.msg):
            exc_info = sys.exc_info()
            if exc_info[1] is not None:
                current = exc_info[1]
                while current is not None:
                    if hasattr(current, "status_code"):
                        code = getattr(current, "status_code", None)
                        body = (
                            getattr(current, "body", None)
                            or getattr(current, "message", None)
                            or ""
                        )
                        record.msg = f"HTTP {code} error — {body}"
                        record.args = ()
                        break
                    current = getattr(current, "__cause__", None)
        return True


# OpenAI SDK logger ("openai") — handles OPENAI_LOG=debug HTTP request/response logs
_openai_logger = logging.getLogger("openai")
_openai_logger.addHandler(_stdout_handler)
_openai_logger.propagate = False

# httpx logger — OPENAI_LOG=debug also enables httpx debug logs
_httpx_logger = logging.getLogger("httpx")
_httpx_logger.addHandler(_stdout_handler)
_httpx_logger.propagate = False

# Agents SDK logger ("openai.agents") — "Error getting response" messages
_agents_logger = logging.getLogger("openai.agents")
_agents_logger.setLevel(logging.ERROR)
_agents_logger.addFilter(_HttpErrorFilter())
_agents_logger.addHandler(_stdout_handler)
_agents_logger.propagate = False

# Auto-discover and register all scenario_*.py modules
from scripts.simulation_data import discover_scenarios  # noqa: E402

discover_scenarios()

from agentic_fraud_servicing.config import get_settings  # noqa: E402
from agentic_fraud_servicing.evaluation.models import EvaluationRun, TurnMetric  # noqa: E402
from agentic_fraud_servicing.evaluation.report import (  # noqa: E402
    extract_dimension_score,
    generate_report,
    save_report,
    save_run,
)
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


def _build_evaluation_run_from_db(
    scenario_name: str,
    data_dir: str,
    transcript_path: str,
    case_id: str,
) -> EvaluationRun:
    """Reconstruct an EvaluationRun from simulation DB + transcript file.

    Reads:
    - traces.db: copilot_suggestion records (hypothesis scores, risk flags,
      suggested questions, latency) and copilot_final state
    - evidence.db: AllegationStatement nodes for extracted allegations
    - transcript file: original turn text (unredacted)
    - scenario module: ground truth

    Returns:
        EvaluationRun ready for the 8-dimension evaluators.
    """
    gateway = create_gateway(data_dir)

    # 1. Load transcript for turn structure and text
    with open(transcript_path) as f:
        transcript_json = f.read()
    events = parse_transcript_json(transcript_json)

    # 2. Load traces — build lookup of suggestion data by turn number
    traces = gateway.trace_store.get_traces_by_case(case_id)

    suggestion_by_turn: dict[int, dict] = {}
    latency_by_turn: dict[int, float] = {}
    copilot_final_state: dict = {}

    for trace in traces:
        if trace["agent_id"] == "copilot_suggestion" and trace["action"] == "suggestion":
            input_data = json.loads(trace["input_data"])
            turn_num = input_data.get("turn")
            if turn_num is not None:
                suggestion_by_turn[turn_num] = json.loads(trace["output_data"])
                latency_by_turn[turn_num] = trace["duration_ms"]
        elif trace["agent_id"] == "copilot_final" and trace["action"] == "final_state":
            copilot_final_state = json.loads(trace["output_data"])

    # 3. Load allegations from evidence store
    all_nodes = gateway.evidence_store.get_nodes_by_case(case_id)
    allegation_nodes = [
        n
        for n in all_nodes
        if n.get("source_type") == "ALLEGATION" or n.get("node_type") == "ALLEGATION_STATEMENT"
    ]

    # 4. Build TurnMetrics from transcript + trace data
    turn_metrics: list[TurnMetric] = []
    total_latency_ms = 0.0

    # Track which assessed turn gets the allegations (put them on the first assessed turn
    # that has a suggestion, since evaluators just collect unique detail_types across all turns)
    allegations_assigned = False

    for i, event in enumerate(events, 1):
        suggestion = suggestion_by_turn.get(i)
        latency_ms = latency_by_turn.get(i, 0.0)
        total_latency_ms += latency_ms

        # Extract hypothesis scores from suggestion or use empty
        hypothesis_scores: dict[str, float] = {}
        if suggestion is not None:
            hypothesis_scores = suggestion.get("hypothesis_scores", {})
        elif copilot_final_state:
            # Non-assessed turns: carry forward from final state (approximate)
            hypothesis_scores = copilot_final_state.get("hypothesis_scores", {})

        # Assign allegations to first assessed turn
        allegations_extracted: list[dict] = []
        if suggestion is not None and not allegations_assigned and allegation_nodes:
            allegations_extracted = [
                {"detail_type": n.get("detail_type", ""), "description": n.get("text", "")}
                for n in allegation_nodes
            ]
            allegations_assigned = True

        metric = TurnMetric(
            turn_number=i,
            speaker=event.speaker.value,
            text=event.text,
            latency_ms=latency_ms,
            copilot_suggestion=suggestion,
            hypothesis_scores=hypothesis_scores,
            allegations_extracted=allegations_extracted,
        )
        turn_metrics.append(metric)

    # 5. Build EvaluationRun
    ground_truth = _load_ground_truth(scenario_name)

    # Use trace timestamps for start/end if available
    start_time = traces[0]["timestamp"] if traces else datetime.now(timezone.utc).isoformat()
    end_time = traces[-1]["timestamp"] if traces else datetime.now(timezone.utc).isoformat()

    return EvaluationRun(
        scenario_name=scenario_name,
        ground_truth=ground_truth,
        turn_metrics=turn_metrics,
        total_turns=len(turn_metrics),
        total_latency_ms=total_latency_ms,
        start_time=start_time,
        end_time=end_time,
        copilot_final_state=copilot_final_state,
    )


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


async def run_evaluation(scenario_name: str, data_dir: str, transcript_path: str) -> None:
    """Load simulation results from DB and run 8-dimension quality evaluators."""
    wall_start = time.perf_counter()

    scenario = get_scenario(scenario_name)
    ground_truth = _load_ground_truth(scenario_name)

    # -- Banner --
    print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  AMEX Fraud Servicing — Evaluation Runner{RESET}")
    print(f"{BOLD}{CYAN}  (Evaluate from Simulation DB){RESET}")
    print(f"{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"\n{BOLD}Scenario: {scenario.title}{RESET}")
    print(f"{DIM}{scenario.description}{RESET}")
    print(f"{DIM}Data dir: {data_dir}{RESET}")
    print(f"{DIM}Transcript: {transcript_path}{RESET}")
    if ground_truth:
        gt_cat = ground_truth.get("investigation_category", "?")
        print(f"{DIM}Ground truth category: {gt_cat}{RESET}")

    # -- Setup model provider (for LLM-powered evaluators) --
    try:
        settings = get_settings()
    except Exception as exc:
        print(f"\n{RED}Error loading settings: {exc}{RESET}")
        print(f"{RED}Ensure .env is configured with LLM_PROVIDER, CONNECTCHAIN_MODEL_INDEX, etc.{RESET}")
        sys.exit(1)

    try:
        model_provider = get_model_provider(settings)
    except Exception as exc:
        print(f"\n{RED}Error creating model provider: {exc}{RESET}")
        print(f"{RED}Check ConnectChain configuration.{RESET}")
        sys.exit(1)

    # ===================================================================
    # Phase 1: Load simulation results from DB
    # ===================================================================
    print(f"\n{BOLD}{GREEN}{'=' * 60}{RESET}")
    print(f"{BOLD}{GREEN}Phase 1: Loading Simulation Results{RESET}")
    print(f"{BOLD}{GREEN}{'=' * 60}{RESET}")

    evaluation_run = _build_evaluation_run_from_db(
        scenario_name,
        data_dir,
        transcript_path,
        scenario.case_id,
    )

    assessed_count = sum(
        1 for m in evaluation_run.turn_metrics if m.copilot_suggestion is not None
    )
    print(f"  Total turns: {evaluation_run.total_turns}")
    print(f"  Assessed turns: {assessed_count}")
    print(f"  Total latency: {evaluation_run.total_latency_ms:.0f}ms")

    # ===================================================================
    # Phase 2: Generate Evaluation Report (8-dimension quality assessment)
    # ===================================================================
    print(f"\n{BOLD}{GREEN}{'=' * 60}{RESET}")
    print(f"{BOLD}{GREEN}Phase 2: Generating Evaluation Report{RESET}")
    print(f"{BOLD}{GREEN}{'=' * 60}{RESET}")

    output_dir = f"data/evaluations/{scenario_name}"
    save_run(evaluation_run, output_dir)
    report = None
    report_path = None
    try:
        report = await generate_report(evaluation_run, model_provider)
        report_path = save_report(report, output_dir)
        print(f"  Report saved to: {report_path}")
    except Exception as exc:
        from agentic_fraud_servicing.copilot.langfuse_tracing import extract_http_error

        status_code, error_body = extract_http_error(exc)
        if status_code is not None:
            print(
                f"  {YELLOW}Warning: Report generation failed (HTTP {status_code}): {error_body[:300]}{RESET}"
            )
        else:
            print(f"  {YELLOW}Warning: Report generation failed: {exc}{RESET}")

    # -- Summary --
    print(f"\n{BOLD}{GREEN}{'=' * 60}{RESET}")
    print(f"{BOLD}{GREEN}Evaluation Summary{RESET}")
    print(f"{BOLD}{GREEN}{'=' * 60}{RESET}")
    print(f"  Total turns: {evaluation_run.total_turns}")
    print(f"  Assessed turns: {assessed_count}")
    print(f"  Total latency: {evaluation_run.total_latency_ms:.0f}ms")
    avg_latency = evaluation_run.total_latency_ms / max(assessed_count, 1)
    print(f"  Avg latency per assessed turn: {avg_latency:.0f}ms")

    final_scores = evaluation_run.copilot_final_state.get("hypothesis_scores", {})
    if final_scores:
        print(f"  Final hypothesis scores: {json.dumps(final_scores, indent=2)}")

    if ground_truth:
        print(f"  Ground truth: {json.dumps(ground_truth, indent=2)}")
    else:
        print(f"  {YELLOW}No ground truth defined for this scenario{RESET}")

    # Print report scores if available
    if report is not None:
        print(f"\n  {BOLD}Overall Quality Score: {report.overall_score:.2f}{RESET}")
        print(f"  {BOLD}Per-Dimension Scores:{RESET}")
        _dim_names = {
            "latency": ("Latency Compliance", report.latency),
            "prediction": ("Prediction Accuracy", report.prediction),
            "question_adherence": ("Question Adherence", report.question_adherence),
            "allegation_quality": ("Allegation Quality", report.allegation_quality),
            "evidence_utilization": ("Evidence Utilization", report.evidence_utilization),
            "convergence": ("Convergence Speed", report.convergence),
            "risk_flag_timeliness": ("Risk Flag Timeliness", report.risk_flag_timeliness),
            "decision_explanation": ("Decision Explanation", report.decision_explanation),
            "note_alignment": ("CCP Note Alignment", report.note_alignment),
        }
        for dim_key, (label, result) in _dim_names.items():
            score = extract_dimension_score(dim_key, result)
            if score is not None:
                color = GREEN if score >= 0.7 else YELLOW if score >= 0.4 else RED
                print(f"    {label:25s} {color}{score:.2f}{RESET}")
            else:
                print(f"    {label:25s} {DIM}N/A{RESET}")

    elapsed = time.perf_counter() - wall_start
    minutes, seconds = divmod(elapsed, 60)
    print(f"\n{BOLD}{GREEN}Evaluation complete.{RESET}")
    print(f"  Elapsed time: {int(minutes)}m {seconds:.1f}s")
    if report_path:
        print(f"  Report saved to: {report_path}")


def main() -> None:
    """Parse CLI args and run the evaluation."""
    parser = argparse.ArgumentParser(
        description="AMEX Fraud Servicing — Evaluation Runner (from Simulation DB)"
    )
    parser.add_argument(
        "--scenario",
        "-s",
        help=f"Scenario to evaluate. Available: {', '.join(list_scenarios())}",
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        help="Path to simulation data directory containing SQLite DBs "
        "(default: data/simulation/{scenario})",
    )
    parser.add_argument(
        "--transcript",
        "-t",
        help="Path to transcript file for turn text "
        "(default: scripts/transcripts/{scenario}.json)",
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
            sim_dir = f"data/simulation/{name}"
            has_db = os.path.isdir(sim_dir)
            print(f"  {name:30s} — {s.title}  (simulation data: {'yes' if has_db else 'no'})")
        return

    if not args.scenario:
        parser.error("--scenario is required (or use --list to see available scenarios)")

    scenario_name = args.scenario
    get_scenario(scenario_name)  # Validate scenario exists

    # Determine data directory (simulation output)
    data_dir = args.data_dir or f"data/simulation/{scenario_name}"
    if not os.path.isdir(data_dir):
        print(f"{RED}Error: Simulation data directory not found: {data_dir}{RESET}")
        print(
            f"{YELLOW}Run the simulation first: python scripts/run_simulation.py -s {scenario_name}{RESET}"
        )
        sys.exit(1)

    # Determine transcript path
    transcript_path = args.transcript or os.path.join(
        "scripts", "transcripts", f"{scenario_name}.json"
    )
    if not os.path.isfile(transcript_path):
        print(f"{RED}Error: Transcript file not found: {transcript_path}{RESET}")
        sys.exit(1)

    # Save output to scenario-specific file
    output_dir = f"data/evaluations/{scenario_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "evaluation_output.txt")

    with open(output_path, "w") as log_file:
        original_stdout = sys.stdout
        sys.stdout = TeeWriter(original_stdout, log_file)
        try:
            asyncio.run(run_evaluation(scenario_name, data_dir, transcript_path))
        finally:
            sys.stdout = original_stdout

    print(f"\nOutput saved to {output_path}")


if __name__ == "__main__":
    main()
