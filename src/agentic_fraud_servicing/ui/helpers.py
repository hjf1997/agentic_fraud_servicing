"""Shared UI helpers for CLI and Gradio interfaces.

Thin wrappers that wire up storage, gateway, provider, and ingestion layers
to avoid duplicating setup code between CLI and Gradio.
"""

from pathlib import Path

from agentic_fraud_servicing.config import get_settings
from agentic_fraud_servicing.gateway.tool_gateway import ToolGateway
from agentic_fraud_servicing.ingestion.transcript import parse_transcript_json
from agentic_fraud_servicing.investigator.case_writer import CasePack
from agentic_fraud_servicing.models.case import CopilotSuggestion
from agentic_fraud_servicing.models.transcript import TranscriptEvent
from agentic_fraud_servicing.providers.base import ModelProvider, get_model_provider
from agentic_fraud_servicing.storage.case_store import CaseStore
from agentic_fraud_servicing.storage.evidence_store import EvidenceStore
from agentic_fraud_servicing.storage.trace_store import TraceStore


def create_gateway(db_dir: str | Path) -> ToolGateway:
    """Create a ToolGateway with all three stores under db_dir.

    Creates the directory if it does not exist. Each store gets its own
    SQLite database file within db_dir.

    Args:
        db_dir: Directory for SQLite database files.

    Returns:
        A ToolGateway wired to CaseStore, EvidenceStore, and TraceStore.
    """
    db_path = Path(db_dir)
    db_path.mkdir(parents=True, exist_ok=True)

    case_store = CaseStore(db_path / "cases.db")
    evidence_store = EvidenceStore(db_path / "evidence.db")
    trace_store = TraceStore(db_path / "traces.db")

    return ToolGateway(case_store, evidence_store, trace_store)


def create_provider() -> ModelProvider:
    """Create the LLM model provider from environment settings.

    Reads LLM_PROVIDER from .env and instantiates the appropriate provider
    (OpenAI or Bedrock).

    Returns:
        A configured ModelProvider instance.
    """
    settings = get_settings()
    return get_model_provider(settings)


def load_transcript_file(path: str | Path) -> list[TranscriptEvent]:
    """Load and parse a transcript JSON file with automatic PII redaction.

    Reads the JSON file at path and passes it through parse_transcript_json,
    which handles both single events and arrays and applies PII redaction.

    Args:
        path: Path to a JSON file containing transcript event(s).

    Returns:
        List of TranscriptEvent models with redacted text.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the JSON is invalid or events are malformed.
    """
    file_path = Path(path)
    json_str = file_path.read_text(encoding="utf-8")
    return parse_transcript_json(json_str)


def format_suggestion_json(suggestion: CopilotSuggestion) -> str:
    """Format a CopilotSuggestion as indented JSON.

    Args:
        suggestion: The copilot suggestion to format.

    Returns:
        JSON string with 2-space indentation.
    """
    return suggestion.model_dump_json(indent=2)


def format_case_pack_json(case_pack: CasePack) -> str:
    """Format a CasePack as indented JSON.

    Args:
        case_pack: The investigation case pack to format.

    Returns:
        JSON string with 2-space indentation.
    """
    return case_pack.model_dump_json(indent=2)
