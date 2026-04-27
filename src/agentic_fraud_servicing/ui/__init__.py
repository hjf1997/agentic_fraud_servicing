"""CLI and Gradio UI layer.

Note: ``create_app`` from ``gradio_app`` is intentionally NOT imported at
module level to avoid eagerly loading the heavy ``gradio`` dependency.
Import it directly when needed::

    from agentic_fraud_servicing.ui.gradio_app import create_app
"""

from agentic_fraud_servicing.ui.cli import main as cli_main
from agentic_fraud_servicing.ui.helpers import (
    create_gateway,
    create_provider,
    format_suggestion_json,
    load_transcript_file,
)

__all__ = [
    "cli_main",
    "create_gateway",
    "create_provider",
    "format_suggestion_json",
    "load_transcript_file",
]
