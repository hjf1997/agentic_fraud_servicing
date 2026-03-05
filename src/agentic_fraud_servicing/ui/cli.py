"""CLI interface for the agentic fraud servicing system.

Provides three subcommands:
  simulate   — Simulate a call by feeding a transcript file through the
               copilot pipeline and printing CopilotSuggestion output.
  investigate — Run the post-call investigator on an existing case and
                print the resulting CasePack.
  view-case  — Look up and display a case from the local database.

Entry point: ``python -m agentic_fraud_servicing.ui.cli``
"""

import argparse
import asyncio
import sys

from agentic_fraud_servicing.copilot.orchestrator import CopilotOrchestrator
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


if __name__ == "__main__":
    main()
