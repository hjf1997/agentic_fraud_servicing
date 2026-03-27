"""CLI interface for the agentic fraud servicing system.

Provides four subcommands:
  simulate   — Simulate a call by feeding a transcript file through the
               copilot pipeline and printing CopilotSuggestion output.
  investigate — Run the post-call investigator on an existing case and
                print the resulting CasePack.
  view-case  — Look up and display a case from the local database.
  evaluate   — Load simulation results from DB, run the 8-dimension
               evaluation, and output the EvaluationReport.

Entry point: ``python -m agentic_fraud_servicing.ui.cli``
"""

import argparse
import asyncio
import os
import sys

from agentic_fraud_servicing.copilot.orchestrator import CopilotOrchestrator
from agentic_fraud_servicing.evaluation.models import EvaluationRun, TurnMetric  # noqa: F401
from agentic_fraud_servicing.evaluation.report import (
    extract_dimension_score,
    generate_report,
    save_report,
)
from agentic_fraud_servicing.ingestion.transcript import parse_transcript_json
from agentic_fraud_servicing.investigator.orchestrator import InvestigatorOrchestrator
from agentic_fraud_servicing.ui.helpers import (
    create_gateway,
    create_provider,
    format_case_pack_json,
    format_suggestion_json,
    load_transcript_file,
)

# -- Text formatters for --output text mode --


def _format_suggestion_text(suggestion) -> str:
    """Format a CopilotSuggestion as structured plain text."""
    lines = [
        f"--- Copilot Suggestion (call={suggestion.call_id}, ts={suggestion.timestamp_ms}) ---",
        f"Summary: {suggestion.running_summary}",
        f"Impersonation Risk: {suggestion.impersonation_risk:.2f}",
    ]
    if suggestion.hypothesis_scores:
        scores = ", ".join(f"{k}: {v:.2f}" for k, v in suggestion.hypothesis_scores.items())
        lines.append(f"Hypothesis Scores: {scores}")
    if suggestion.suggested_questions:
        lines.append("Suggested Questions:")
        for q in suggestion.suggested_questions:
            lines.append(f"  - {q}")
    if suggestion.risk_flags:
        lines.append("Risk Flags:")
        for flag in suggestion.risk_flags:
            lines.append(f"  - {flag}")
    if suggestion.retrieved_facts:
        lines.append("Retrieved Facts:")
        for fact in suggestion.retrieved_facts:
            lines.append(f"  - {fact}")
    if suggestion.safety_guidance:
        lines.append(f"Safety: {suggestion.safety_guidance}")
    return "\n".join(lines)


def _format_case_pack_text(case_pack) -> str:
    """Format a CasePack as structured plain text."""
    lines = [
        "--- Investigation Case Pack ---",
        f"Summary: {case_pack.case_summary}",
    ]
    if case_pack.timeline:
        lines.append("Timeline:")
        for entry in case_pack.timeline:
            ts = entry.get("timestamp", "?")
            desc = entry.get("description", "")
            lines.append(f"  [{ts}] {desc}")
    if case_pack.evidence_list:
        lines.append(f"Evidence Items: {len(case_pack.evidence_list)}")
    if case_pack.decision_recommendation:
        rec = case_pack.decision_recommendation
        lines.append(
            f"Decision: category={rec.get('category', '?')}, "
            f"confidence={rec.get('confidence', '?')}"
        )
    if case_pack.investigation_notes:
        lines.append("Notes:")
        for note in case_pack.investigation_notes:
            lines.append(f"  - {note}")
    return "\n".join(lines)


# -- Subcommand handlers --


async def cmd_simulate(args: argparse.Namespace) -> None:
    """Simulate a fraud call by running transcript events through the copilot."""
    try:
        events = load_transcript_file(args.transcript)
    except FileNotFoundError:
        print(f"Error: transcript file not found: {args.transcript}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"Error: invalid transcript: {exc}", file=sys.stderr)
        sys.exit(1)

    gateway = create_gateway(args.db_dir)
    provider = create_provider()
    orchestrator = CopilotOrchestrator(gateway, provider)

    for event in events:
        suggestion = await orchestrator.process_event(event)
        if args.output == "json":
            print(format_suggestion_json(suggestion))
        else:
            print(_format_suggestion_text(suggestion))
        print()


async def cmd_investigate(args: argparse.Namespace) -> None:
    """Run the post-call investigator on a case."""
    gateway = create_gateway(args.db_dir)
    provider = create_provider()
    orchestrator = InvestigatorOrchestrator(gateway, provider)

    try:
        case_pack = await orchestrator.investigate(args.case_id)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.output == "json":
        print(format_case_pack_json(case_pack))
    else:
        print(_format_case_pack_text(case_pack))


def cmd_view_case(args: argparse.Namespace) -> None:
    """View case details from the local database."""
    gateway = create_gateway(args.db_dir)
    case = gateway.case_store.get_case(args.case_id)

    if case is None:
        print(f"Error: case not found: {args.case_id}", file=sys.stderr)
        sys.exit(1)

    print(case.model_dump_json(indent=2))


def _format_report_text(report) -> str:
    """Format an EvaluationReport as structured plain text."""
    lines = [
        "--- Evaluation Report ---",
        f"Scenario: {report.scenario_name}",
        f"Overall Score: {report.overall_score:.2f}",
        "",
        "Per-Dimension Scores:",
    ]
    _dims = {
        "latency": ("Latency Compliance", report.latency),
        "prediction": ("Prediction Accuracy", report.prediction),
        "question_adherence": ("Question Adherence", report.question_adherence),
        "allegation_quality": ("Allegation Quality", report.allegation_quality),
        "evidence_utilization": ("Evidence Utilization", report.evidence_utilization),
        "convergence": ("Convergence Speed", report.convergence),
        "risk_flag_timeliness": ("Risk Flag Timeliness", report.risk_flag_timeliness),
        "decision_explanation": ("Decision Explanation", report.decision_explanation),
    }
    for dim_key, (label, result) in _dims.items():
        score = extract_dimension_score(dim_key, result)
        if score is not None:
            lines.append(f"  {label:25s} {score:.2f}")
        else:
            lines.append(f"  {label:25s} N/A")

    # Key metrics
    lines.append("")
    lines.append("Key Metrics:")
    if report.prediction is not None:
        match_str = "YES" if report.prediction.match else "NO"
        lines.append(f"  Prediction Match: {match_str}")
    if report.convergence is not None:
        turn = report.convergence.convergence_turn
        lines.append(f"  Convergence Turn: {turn if turn is not None else 'never'}")
    if report.latency is not None:
        lines.append(f"  Latency P95: {report.latency.p95_ms:.0f}ms")
    if report.allegation_quality is not None:
        lines.append(f"  Allegation F1: {report.allegation_quality.f1_score:.2f}")

    return "\n".join(lines)


def _ensure_scripts_importable() -> None:
    """Ensure the ``scripts`` package is importable by adding the project root to sys.path."""
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


async def cmd_evaluate(args: argparse.Namespace) -> None:
    """Load simulation results from DB and produce an evaluation report."""
    import importlib
    import json
    from datetime import datetime, timezone

    # Ensure scripts package is importable (scenario modules live there)
    _ensure_scripts_importable()

    # Auto-discover and register all scenario_*.py modules
    from scripts.simulation_data import discover_scenarios, get_scenario

    discover_scenarios()

    # Load scenario
    try:
        scenario = get_scenario(args.scenario)
    except KeyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # Determine paths
    db_dir = args.db_dir or f"data/simulation/{args.scenario}"
    transcript_path = args.transcript
    if transcript_path is None:
        transcript_path = os.path.join("scripts", "transcripts", f"{args.scenario}.json")

    if not os.path.isfile(transcript_path):
        print(f"Error: transcript file not found: {transcript_path}", file=sys.stderr)
        sys.exit(1)

    # Load simulation data from DB + transcript
    gateway = create_gateway(db_dir)
    provider = create_provider()

    # Load transcript for turn text
    events = parse_transcript_json(open(transcript_path).read())

    # Load traces
    traces = gateway.trace_store.get_traces_by_case(scenario.case_id)
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

    # Load allegations from evidence store
    all_nodes = gateway.evidence_store.get_nodes_by_case(scenario.case_id)
    allegation_nodes = [
        n for n in all_nodes
        if n.get("source_type") == "ALLEGATION" or n.get("node_type") == "ALLEGATION_STATEMENT"
    ]

    # Build TurnMetrics
    turn_metrics: list[TurnMetric] = []
    total_latency_ms = 0.0
    allegations_assigned = False

    for i, event in enumerate(events, 1):
        suggestion = suggestion_by_turn.get(i)
        latency_ms = latency_by_turn.get(i, 0.0)
        total_latency_ms += latency_ms

        hypothesis_scores: dict[str, float] = {}
        if suggestion is not None:
            hypothesis_scores = suggestion.get("hypothesis_scores", {})
        elif copilot_final_state:
            hypothesis_scores = copilot_final_state.get("hypothesis_scores", {})

        allegations_extracted: list[dict] = []
        if suggestion is not None and not allegations_assigned and allegation_nodes:
            allegations_extracted = [
                {"detail_type": n.get("detail_type", ""), "description": n.get("text", "")}
                for n in allegation_nodes
            ]
            allegations_assigned = True

        turn_metrics.append(TurnMetric(
            turn_number=i,
            speaker=event.speaker.value,
            text=event.text,
            latency_ms=latency_ms,
            copilot_suggestion=suggestion,
            hypothesis_scores=hypothesis_scores,
            allegations_extracted=allegations_extracted,
        ))

    # Load ground truth
    ground_truth: dict = {}
    try:
        mod = importlib.import_module(f"scripts.scenario_{args.scenario}")
        ground_truth = getattr(mod, "GROUND_TRUTH", {})
    except (ImportError, AttributeError):
        pass

    start_time = traces[0]["timestamp"] if traces else datetime.now(timezone.utc).isoformat()
    end_time = traces[-1]["timestamp"] if traces else datetime.now(timezone.utc).isoformat()

    evaluation_run = EvaluationRun(
        scenario_name=args.scenario,
        ground_truth=ground_truth,
        turn_metrics=turn_metrics,
        total_turns=len(turn_metrics),
        total_latency_ms=total_latency_ms,
        start_time=start_time,
        end_time=end_time,
        copilot_final_state=copilot_final_state,
    )

    # Generate report
    report = await generate_report(evaluation_run, provider)

    output_dir = f"data/evaluations/{args.scenario}"
    os.makedirs(output_dir, exist_ok=True)
    save_report(report, output_dir)

    # Output
    if args.output == "json":
        print(report.model_dump_json(indent=2))
    else:
        print(_format_report_text(report))


# -- Argument parser --


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="agentic_fraud_servicing.ui.cli",
        description="Agentic fraud servicing CLI — simulate calls, investigate cases, view data.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # simulate
    sim = subparsers.add_parser("simulate", help="Simulate a call with a transcript file")
    sim.add_argument("-t", "--transcript", required=True, help="Path to JSON transcript file")
    sim.add_argument("-d", "--db-dir", default="data/cli", help="Directory for SQLite databases")
    sim.add_argument(
        "-o", "--output", choices=["json", "text"], default="json", help="Output format"
    )

    # investigate
    inv = subparsers.add_parser("investigate", help="Run post-call investigation on a case")
    inv.add_argument("-c", "--case-id", required=True, help="Case ID to investigate")
    inv.add_argument("-d", "--db-dir", default="data/cli", help="Directory for SQLite databases")
    inv.add_argument(
        "-o", "--output", choices=["json", "text"], default="json", help="Output format"
    )

    # view-case
    vc = subparsers.add_parser("view-case", help="View case details from the database")
    vc.add_argument("-c", "--case-id", required=True, help="Case ID to view")
    vc.add_argument("-d", "--db-dir", default="data/cli", help="Directory for SQLite databases")

    # evaluate
    ev = subparsers.add_parser("evaluate", help="Evaluate simulation results for a scenario")
    ev.add_argument("-s", "--scenario", required=True, help="Scenario name to evaluate")
    ev.add_argument("-t", "--transcript", default=None, help="Override transcript file path")
    ev.add_argument("-d", "--db-dir", default=None, help="Simulation data directory with SQLite DBs")
    ev.add_argument(
        "-o", "--output", choices=["json", "text"], default="json", help="Output format"
    )

    return parser


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate handler."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "simulate":
        asyncio.run(cmd_simulate(args))
    elif args.command == "investigate":
        asyncio.run(cmd_investigate(args))
    elif args.command == "view-case":
        cmd_view_case(args)
    elif args.command == "evaluate":
        asyncio.run(cmd_evaluate(args))


if __name__ == "__main__":
    main()
