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

## Evaluation

The evaluation layer replays pre-existing transcripts through the copilot
pipeline and measures quality across 8 dimensions against known ground truth
outcomes. This enables enterprise-grade assessment of copilot performance on
real or synthetic fraud cases.

### Running an Evaluation

```bash
# List available evaluation scenarios
python scripts/run_evaluation.py --list

# Run evaluation for a specific scenario (replays transcript + generates report)
python scripts/run_evaluation.py --scenario <name>

# Or use the CLI subcommand
python -m agentic_fraud_servicing.ui.cli evaluate --scenario <name>
python -m agentic_fraud_servicing.ui.cli evaluate --scenario <name> --output text
```

### Viewing Results

Evaluation results can be viewed in three ways:

```bash
# Interactive Gradio dashboard (AMEX-branded, no LLM credentials needed)
python -m agentic_fraud_servicing.ui.eval_dashboard

# Static self-contained HTML report
python scripts/export_eval_report.py --scenario <name>

# Raw JSON data
# data/evaluations/{scenario}/evaluation_run.json     (per-turn metrics)
# data/evaluations/{scenario}/evaluation_report.json  (8-dimension scores)
```

### Evaluation Dimensions

| Dimension | Description |
|-----------|-------------|
| Latency Compliance | Per-turn wall-clock time vs 1500ms target (p50/p95/p99) |
| Prediction Accuracy | Copilot's highest hypothesis vs ground truth InvestigationCategory |
| Question Adherence | Whether the CCP incorporated suggested questions |
| Allegation Extraction Quality | Precision, recall, and F1 of extracted allegations |
| Evidence Utilization | Retrieval and reasoning coverage of available evidence |
| Convergence Speed | Turn at which the correct category becomes and stays dominant |
| Risk Flag Timeliness | When flags were raised vs when evidence became available |
| Decision Explanation | Reasoning chain quality, influential evidence, and improvement suggestions |

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
    evaluation/             # 8-dimension copilot quality evaluators and report
    ui/                     # CLI, Gradio web interface, and dashboards
scripts/
    sample_transcript.json  # Sample fraud call transcript for testing
    run_simulation.py       # Full-scale E2E simulation with live LLM
    run_evaluation.py       # Evaluation runner (transcript replay + reporting)
    export_eval_report.py   # Static HTML report exporter for evaluations
docs/
    policies/               # Policy documents for Case Advisor agent
tests/
    unit/                   # Unit tests (mocked LLM calls)
    integration/            # Integration tests (end-to-end flows)
```
