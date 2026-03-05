# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

---

## Project Overview

Banking-grade multi-agent fraud servicing system for AMEX card disputes. Provides
two capabilities: (1) Realtime Copilot — interprets live transcript streams during
calls, tracks fraud/dispute/scam hypotheses, suggests next-best questions, and
assesses impersonation risk; (2) Post-Call Investigator — builds evidence graphs,
maps scheme reason codes, detects scam patterns and contradictions, and generates
auditable case packs with decision recommendations. Uses OpenAI Agents SDK for
hub-and-spoke orchestration with a provider abstraction layer supporting Claude
via AWS Bedrock (dev) and OpenAI (enterprise).

---

## Commit Rules

- Do not include "claude", "anthropic", or "AI" in commit messages
- Do not include `Co-Authored-By` lines referencing AI tools
- Format:
  ```
  <type>(<scope>): <subject line - max 50-72 chars>

  <why paragraph - 2-4 sentences explaining context and problem>

  <how paragraph - 2-4 sentences explaining solution at high level>

  CHANGE: <1-2 sentences summarizing what changed>
  ```

---

## Code Conventions

### Module Structure and Imports

- All source code lives under `src/agentic_fraud_servicing/`
- Use absolute imports from the package root: `from agentic_fraud_servicing.models.case import Case`
- Dependency direction is strictly one-way:
  - `ui/` -> `copilot/`, `investigator/` -> `gateway/` -> `storage/`
  - `models/` and `config.py` are leaf dependencies — imported by all, import nothing from the project
  - `ingestion/` feeds into `copilot/` only
  - Agents (`copilot/`, `investigator/`) never import from `storage/` directly — all data access goes through `gateway/`
- Each `__init__.py` re-exports the module's public API (key classes/functions)

### Data Models and Validation

- All domain models (Case, Evidence, TranscriptEvent, etc.) are Pydantic v2 `BaseModel` subclasses
- Validate at system boundaries only: ingestion input, Tool Gateway request/response, UI input
- Internal function-to-function calls pass typed Pydantic models — no redundant re-validation
- Enums live in `models/enums.py` and are used for all categorical fields (speaker type, risk level, case status, evidence edge type)
- Serialization: use `.model_dump()` for dict conversion, `.model_dump_json()` for JSON — never `json.dumps()` on Pydantic models
- Evidence nodes must have a `source_type` distinguishing FACT vs ALLEGATION — never mix verified system data with customer claims

### LLM Provider Abstraction

- All LLM calls go through the provider interface in `providers/base.py`
- Never call `boto3` or `openai` directly outside the provider implementations
- Provider selection is determined by `LLM_PROVIDER` env var (`openai` or `bedrock`)
- Bedrock provider uses `boto3.client("bedrock-runtime").converse()` — not `invoke_model()`
- Bedrock model ID: `us.anthropic.claude-sonnet-4-5-20250929-v1:0`, region: `us-east-1`
- Agent orchestration always uses OpenAI Agents SDK patterns (tool registration, agent-as-tool) regardless of LLM backend

### Agent-as-Tool Pattern

- Specialist agents are registered as tools on the orchestrator agent — never use free handoffs
- Each agent tool call is a single function: takes a typed input, returns a typed output
- Orchestrators (`copilot/orchestrator.py`, `investigator/orchestrator.py`) maintain the control loop and decide which agent runs when
- Every agent invocation must be logged to the Trace Store with input, output, and duration

### Tool Gateway

- All data access (reads and writes) goes through `gateway/tool_gateway.py`
- Gateway enforces: authentication check, field-level masking (no raw PAN), rate limiting, purpose limitation, immutable audit logging
- Read tools in `gateway/tools/read_tools.py`: transaction lookups, auth log queries, profile fetches
- Write tools in `gateway/tools/write_tools.py`: case creation, evidence append
- Compliance tools in `gateway/tools/compliance.py`: retention checks, consent verification, redaction utilities
- Agents never import from `storage/` — they call gateway tools

### Storage Layer

- SQLite via stdlib `sqlite3` — no ORM
- Case data scoped to `data/cases/{case_id}/`
- Trace/audit logs to `data/traces/`
- SQLite uses WAL mode for concurrent read support
- Storage classes (`case_store.py`, `evidence_store.py`, `trace_store.py`) are the only modules that touch the database
- The `data/` directory is gitignored

### Ingestion and Redaction

- All transcript text must pass through `ingestion/redaction.py` before reaching any LLM or storage
- Redaction covers: PAN (card numbers), CVV, SSN, DOB, physical addresses
- Redacted values are replaced with typed placeholders: `[PAN_REDACTED]`, `[CVV_REDACTED]`, etc.
- Raw PAN/CVV is never persisted anywhere — not in logs, not in evidence, not in traces

### Error Handling

- Use `ValueError` for invalid input at system boundaries (bad transcript format, missing required fields)
- Use `RuntimeError` for system failures (database errors, provider connection failures)
- Use domain-specific exceptions sparingly: `RedactionError` for redaction pipeline failures, `GatewayAuthError` for auth failures
- All gateway tool calls must catch and log exceptions before re-raising — never let raw database errors reach agents
- LLM provider errors should be caught in the provider layer and wrapped with context (model ID, request type)

### Async Conventions

- Copilot agents use `async` — latency-sensitive path (< 1.5s target)
- Post-call investigator agents may use sync or async — no strict latency requirement
- Use `httpx.AsyncClient` for any HTTP calls within async code paths
- Never use `asyncio.run()` inside an already-running event loop — use `await` instead

### Configuration

- All config loaded via `config.py` using `python-dotenv`
- Required env vars: `LLM_PROVIDER`, `AWS_PROFILE` (for bedrock), `AWS_REGION`, `AWS_BEDROCK_MODEL_ID`
- Optional: `OPENAI_API_KEY` (only when `LLM_PROVIDER=openai`)
- `.env.example` has placeholder values for all vars — keep it updated when adding new config
- Never hardcode credentials, model IDs, or region strings outside `.env` and `config.py`

### Testing

- All tests in `tests/` with `unit/` and `integration/` subdirs
- Default: all LLM calls and external services are mocked (fixtures in `conftest.py`)
- Use `@pytest.mark.live` for integration tests that call real Bedrock endpoints
- Run unit tests: `pytest tests/ -v` (excludes live-marked tests by default)
- Run live tests: `pytest tests/ -v -m live`
- Mock data created per-test, cleaned up in teardown — no persistent test fixtures
- Test file naming: `test_<module_name>.py` matching the source module
- Use `pytest-asyncio` for async test functions

### CLI and UI

- CLI via `ui/cli.py` using `argparse` — supports simulating call transcripts and triggering investigation
- Gradio app via `ui/gradio_app.py` — streaming web demo
- Both UIs call the same orchestrator interfaces — no business logic in UI layer
- CLI output is structured (JSON or formatted text) — never raw debug prints in production paths

---

## Rules for Python Comments

- **Rule of Thumb**:
  - Every `.py` should have a header comment or module docstring unless intent is obvious
  - Every function/class should have a docstring
  - Every 20-30 lines of logic should have at least one comment explaining intent
  - For every tricky block, explain the `why`, not `what`
- **When you should comment**:
  - Non-obvious logic: explain why something is done
  - Complex algorithm: summarize approach in plain language
  - Function/Module purpose: describe role, inputs, outputs, and side effects
- **When you should not comment**:
  - Do not state the obvious
  - Do not duplicate docstrings or variable names

---

## Rules for Python Type Hints

- Always have type hints for function parameters and return types
- Often for dataclass/class attributes
- Sometimes for tricky variables where type is not obvious
- Never for obvious local variables

---

## Important Rules

- Apply YAGNI principle (You Aren't Gonna Need It)
- Keep code clean and scalable
- Provide clean, maintainable fixes instead of minimal patches
- Avoid naming conventions like "enhanced", "new", "latest", "better", "best"

---

## Build and Development Commands

```bash
# Activate environment
conda activate agentic_fraud_servicing

# Run linting
ruff check . --fix

# Run formatting
ruff format .

# Run pre-commit hooks
pre-commit run --all-files

# Run tests
pytest tests/ -v
```
