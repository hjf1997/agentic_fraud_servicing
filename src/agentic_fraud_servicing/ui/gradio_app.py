"""Gradio web demo for the agentic fraud servicing system.

Provides a single-tab interface for Realtime Copilot Simulation: paste
transcript JSON, process events through the copilot pipeline, and view
suggestions.

Entry point: ``python -m agentic_fraud_servicing.ui.gradio_app``
"""

import gradio as gr

from agentic_fraud_servicing.copilot.orchestrator import CopilotOrchestrator
from agentic_fraud_servicing.ingestion.transcript import parse_transcript_json
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

        # Find the last CARDMEMBER event — is_last must target CM events
        # because non-CM events return None before is_last is checked.
        last_cm_idx = 0
        for idx, ev in enumerate(events, 1):
            if ev.speaker == "CARDMEMBER":
                last_cm_idx = idx

        suggestions = []
        for i, event in enumerate(events, 1):
            suggestion = await orchestrator.process_event(event, is_last=(i == last_cm_idx))
            if suggestion is not None:
                suggestions.append(suggestion.model_dump(mode="json"))
        return suggestions
    except Exception as exc:
        return [{"error": str(exc)}]


def create_app() -> gr.Blocks:
    """Create the Gradio Blocks app for copilot simulation.

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

    return app


def main() -> None:
    """Launch the Gradio web demo."""
    app = create_app()
    app.launch()


if __name__ == "__main__":
    main()
