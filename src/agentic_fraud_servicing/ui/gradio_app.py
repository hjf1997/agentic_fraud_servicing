"""Gradio web demo for the agentic fraud servicing system.

Provides a two-tab interface:
  Tab 1 — Realtime Copilot Simulation: paste transcript JSON, process events
           through the copilot pipeline, and view suggestions.
  Tab 2 — Post-Call Investigation: enter a case ID, run the investigator
           pipeline, and view the resulting case pack.

Entry point: ``python -m agentic_fraud_servicing.ui.gradio_app``
"""

import gradio as gr

from agentic_fraud_servicing.copilot.orchestrator import CopilotOrchestrator
from agentic_fraud_servicing.ingestion.transcript import parse_transcript_json
from agentic_fraud_servicing.investigator.orchestrator import InvestigatorOrchestrator
from agentic_fraud_servicing.ui.helpers import create_gateway, create_provider


async def process_transcript(transcript_json: str, db_dir: str) -> list[dict]:
    """Parse transcript JSON through the copilot pipeline.

    Parses the JSON through the ingestion/redaction pipeline, feeds each event
    to a CopilotOrchestrator, and returns the list of suggestion dicts.

    Args:
        transcript_json: JSON string containing transcript event(s).
        db_dir: Directory for SQLite database files.

    Returns:
        List of CopilotSuggestion dicts, or a single-element error list.
    """
    try:
        events = parse_transcript_json(transcript_json)
        gateway = create_gateway(db_dir)
        provider = create_provider()
        orchestrator = CopilotOrchestrator(gateway, provider)

        suggestions = []
        for event in events:
            suggestion = await orchestrator.process_event(event)
            suggestions.append(suggestion.model_dump(mode="json"))
        return suggestions
    except Exception as exc:
        return [{"error": str(exc)}]


async def run_investigation(case_id: str, db_dir: str) -> dict:
    """Run the post-call investigator on a case.

    Creates an InvestigatorOrchestrator and runs the investigation for the
    given case_id.

    Args:
        case_id: The case to investigate.
        db_dir: Directory for SQLite database files.

    Returns:
        CasePack dict, or an error dict.
    """
    try:
        gateway = create_gateway(db_dir)
        provider = create_provider()
        orchestrator = InvestigatorOrchestrator(gateway, provider)
        case_pack = await orchestrator.investigate(case_id)
        return case_pack.model_dump(mode="json")
    except Exception as exc:
        return {"error": str(exc)}


def create_app() -> gr.Blocks:
    """Create the Gradio Blocks app with two tabs.

    Returns:
        A configured gr.Blocks instance ready to launch.
    """
    with gr.Blocks(title="Agentic Fraud Servicing") as app:
        gr.Markdown("# Agentic Fraud Servicing Demo")

        with gr.Tab("Copilot Simulation"):
            transcript_input = gr.Textbox(
                label="Transcript JSON",
                lines=10,
                placeholder="Paste transcript events as a JSON array...",
            )
            copilot_db_dir = gr.Textbox(
                label="DB Directory",
                value="data/gradio",
            )
            copilot_btn = gr.Button("Process Transcript")
            copilot_output = gr.JSON(label="Copilot Suggestions")

            copilot_btn.click(
                fn=process_transcript,
                inputs=[transcript_input, copilot_db_dir],
                outputs=copilot_output,
            )

        with gr.Tab("Post-Call Investigation"):
            case_id_input = gr.Textbox(label="Case ID")
            invest_db_dir = gr.Textbox(
                label="DB Directory",
                value="data/gradio",
            )
            invest_btn = gr.Button("Run Investigation")
            invest_output = gr.JSON(label="Investigation Results")

            invest_btn.click(
                fn=run_investigation,
                inputs=[case_id_input, invest_db_dir],
                outputs=invest_output,
            )

    return app


def main() -> None:
    """Launch the Gradio web demo."""
    app = create_app()
    app.launch()


if __name__ == "__main__":
    main()
