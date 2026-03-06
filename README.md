# Agentic Fraud Servicing

Banking-grade multi-agent fraud servicing system for AMEX card disputes. The
system provides two tightly integrated capabilities: a **Realtime Copilot** that
interprets live transcript streams during calls, tracks fraud/dispute/scam
hypotheses, suggests next-best questions, and assesses impersonation risk; and a
**Post-Call Investigator** that builds evidence graphs, maps scheme reason codes
(AMEX/Visa/Mastercard), detects scam patterns and contradictions, and generates
auditable case packs with decision recommendations.

## Features

- **PCI-compliant PII redaction** — PAN, CVV, SSN, DOB, and address patterns
  redacted before any LLM or storage layer sees the text
- **Dual LLM provider** — Claude via AWS Bedrock (development) or OpenAI GPT-4o
  (enterprise), switchable via a single environment variable
- **Evidence graph with fact/allegation separation** — typed evidence nodes
  distinguish system-verified facts from customer-stated claims
- **Hub-and-spoke agent orchestration** — specialist agents invoked as tools by
  a central orchestrator for full auditability of every invocation
- **Audit trail via Tool Gateway** — all data access goes through a gateway
  layer that enforces authentication, field-level masking, and immutable logging
- **Scheme reason code mapping** — maps allegations to AMEX (C08, FR2, FR4),
  Visa (10.4, 13.1), and Mastercard (4837, 4853) reason codes
- **Scam pattern detection** — identifies APP fraud, romance scams, phishing,
  first-party fraud, and manipulation indicators in transcripts

## Architecture

The system follows a hub-and-spoke multi-agent pattern where central
orchestrators invoke specialist agents as tools. Data flows through a layered
module hierarchy:

```
Ingestion (redaction + parsing)
    -> Copilot / Investigator (agent orchestration)
        -> Tool Gateway (auth, masking, logging)
            -> Storage (SQLite case, evidence, trace stores)
```

The **Realtime Copilot** runs four specialist agents per transcript event:
triage (claim extraction), auth assessment (impersonation risk), question
planner (next-best questions), and fast retrieval (data lookup via gateway).

The **Post-Call Investigator** runs four specialist agents per case: scheme
mapper (reason codes), merchant evidence (normalization and conflicts), scam
detector (contradictions and patterns), and case writer (narrative and
recommendation).

Both use the OpenAI Agents SDK for orchestration with a provider abstraction
layer that translates to either Bedrock `converse()` or OpenAI Chat Completions.

## Setup

### Prerequisites

- Python 3.13+
- Conda (recommended) or any Python environment manager

### Installation

```bash
# Create and activate environment
conda create -n agentic_fraud_servicing python=3.13
conda activate agentic_fraud_servicing

# Install package with development dependencies
pip install -e '.[dev]'
```

### Configuration

Copy the example environment file and configure credentials:

```bash
cp .env.example .env
```

Edit `.env` with your settings:

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM backend: `bedrock` or `openai` | `bedrock` |
| `AWS_PROFILE` | AWS credentials profile for Bedrock access | `default` |
| `AWS_REGION` | AWS region for the Bedrock endpoint | `us-east-1` |
| `AWS_BEDROCK_MODEL_ID` | Bedrock model identifier | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` |
| `OPENAI_API_KEY` | OpenAI API key (required only when `LLM_PROVIDER=openai`) | — |

## Usage

### CLI

The CLI supports three subcommands for simulation and inspection:

```bash
# Simulate a copilot session with a transcript file
python -m agentic_fraud_servicing.ui.cli simulate \
    --transcript scripts/sample_transcript.json

# Run post-call investigation on a case
python -m agentic_fraud_servicing.ui.cli investigate --case-id case-001

# View a stored case
python -m agentic_fraud_servicing.ui.cli view-case --case-id case-001
```

Add `--output text` for human-readable output (default is JSON).

### Gradio Web Demo

Launch the interactive web interface with copilot simulation and investigation
tabs:

```bash
python -m agentic_fraud_servicing.ui.gradio_app
```

## Testing

```bash
# Run all tests (unit + integration, LLM calls mocked)
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=agentic_fraud_servicing

# Run live integration tests against real Bedrock (requires credentials)
pytest tests/ -v -m live

# Lint and format checks
ruff check .
ruff format --check .
```

## Project Structure

```
src/agentic_fraud_servicing/
    config.py               # Settings and environment loading
    models/                 # Pydantic v2 domain models and enums
    providers/              # LLM provider abstraction (OpenAI + Bedrock)
    ingestion/              # Transcript parsing and PII redaction
    copilot/                # Realtime copilot agents and orchestrator
    investigator/           # Post-call investigator agents and orchestrator
    gateway/                # Tool Gateway with auth, masking, and logging
        tools/              # Read, write, and compliance tool functions
    storage/                # SQLite stores for cases, evidence, and traces
    ui/                     # CLI and Gradio web interface
scripts/
    sample_transcript.json  # Sample fraud call transcript for testing
tests/
    unit/                   # Unit tests (mocked LLM calls)
    integration/            # Integration tests (end-to-end flows)
```
