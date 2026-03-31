# Copilot Agent Workflow

The Realtime Copilot processes live transcript events during fraud servicing calls. It uses a hub-and-spoke architecture where the `CopilotOrchestrator` (a plain Python class, not an Agents SDK Agent) explicitly controls which specialist agents run and when.

---

## Overview

```mermaid
flowchart TD
    Event["TranscriptEvent"] --> Orch["CopilotOrchestrator<br/>.process_event()"]

    Orch --> SpeakerCheck{Speaker?}

    SpeakerCheck -->|CCP / SYSTEM| ReturnPrev["Return previous<br/>CopilotSuggestion<br/>(no LLM calls)"]
    SpeakerCheck -->|CARDMEMBER| IntervalCheck

    IntervalCheck{"Assessment turn?<br/>(every assess_interval<br/>CM turns, default=5)"} -->|"No (skip)"| ReturnPrev2["Return previous<br/>CopilotSuggestion<br/>(no LLM calls)"]
    IntervalCheck -->|"Yes (CM turn 1, 6, 11, ...)"| Phase1

    subgraph Phase1["Phase 1: Parallel — asyncio.gather()"]
        direction LR
        Triage["Triage Agent<br/>─────────────<br/>Extract 0-8 allegations<br/>per turn using 17-value<br/>AllegationDetailType"]
        Auth["Auth Agent<br/>─────────────<br/>Impersonation risk<br/>score (0.0-1.0)<br/>+ step-up method"]
        Retrieval["Retrieval Agent<br/>─────────────<br/>Fetch transactions,<br/>auth logs, customer<br/>profile (idempotent)"]
    end

    Phase1 --> StateUpdate["State Updates<br/>─────────────<br/>accumulated_allegations += new<br/>impersonation_risk updated<br/>_retrieval_result cached<br/>missing_fields pruned<br/>evidence_collected updated"]

    subgraph Phase2["Phase 2: Parallel — asyncio.gather()"]
        direction LR
        Hyp["Hypothesis Agent<br/>─────────────<br/>Bayesian scoring:<br/>THIRD_PARTY_FRAUD<br/>FIRST_PARTY_FRAUD<br/>SCAM | DISPUTE"]
        CaseAdv["Case Advisor<br/>─────────────<br/>Policy-aware<br/>eligibility check<br/>+ 0-3 questions<br/>+ stopping signal"]
    end

    StateUpdate --> Phase2
    Phase2 --> Output["CopilotSuggestion<br/>─────────────<br/>suggested_questions<br/>risk_flags<br/>hypothesis_scores<br/>impersonation_risk<br/>case_eligibility<br/>information_sufficient<br/>running_summary<br/>safety_guidance<br/>retrieved_facts"]

    Auth -.- AuthNote["Turns 1-3: always<br/>Turn 4+: only if risk ≥ 0.4"]
    Retrieval -.- RetrNote["Runs once per session<br/>(cached thereafter)"]

    style Phase1 fill:#e8f4fd,stroke:#1a73e8
    style Phase2 fill:#e8f4fd,stroke:#1a73e8
    style Output fill:#e6f4ea,stroke:#1e8e3e
    style ReturnPrev fill:#fce8e6,stroke:#d93025
    style ReturnPrev2 fill:#fce8e6,stroke:#d93025
    style AuthNote fill:#fff3e0,stroke:#f57c00,stroke-dasharray: 5 5
    style RetrNote fill:#fff3e0,stroke:#f57c00,stroke-dasharray: 5 5
```

Only **CARDMEMBER** events trigger the agent pipeline, and only on **assessment turns** — every `assess_interval` CM turns (default 5). CM turns 1, 6, 11, 16, ... run the full pipeline; all other turns return the previous suggestion immediately. CCP and SYSTEM events never trigger the pipeline.

For assessment turns the pipeline is:

1. **Phase 1 (parallel)**: Triage + Auth (conditional) + Retrieval
2. **Phase 2 (parallel, turn 4+)**: Hypothesis + Case Advisor — or Hypothesis only on turns 1-3

---

## Phase 1 — Parallel (`asyncio.gather`)

### 1. Triage Agent

**Role**: Extract structured allegations from cardmember statements.

| Direction | Field | Type | Description |
|-----------|-------|------|-------------|
| **INPUT** | `conversation_history` | `list[(speaker, text)]` | Context + new turns since last assessment (marked [CONTEXT]/[NEW]/[LATEST TURN]) |
| **INPUT** | `new_turn_offset` | `int` | Index where [NEW] turns begin (entries before are [CONTEXT]) |
| **INPUT** | `allegation_summary` | `str \| None` | Previously extracted allegations for dedup (when allegations exist) |
| **INPUT** | `model_provider` | `ModelProvider` | LLM provider for inference |

| Direction | Field | Type | Description |
|-----------|-------|------|-------------|
| **OUTPUT** | `allegations` | `list[AllegationExtraction]` | 0-8 per turn (typically 0-2) |
| | `.detail_type` | `AllegationDetailType` | One of 17 values (e.g. `UNRECOGNIZED_TRANSACTION`, `LOST_STOLEN_CARD`, `GOODS_NOT_RECEIVED`, `DUPLICATE_CHARGE`) |
| | `.description` | `str` | Paraphrase of what the CM alleged |
| | `.entities` | `dict[str, str]` | Structured key-value pairs (e.g. `merchant_name`, `amount`, `transaction_date`) |
| | `.confidence` | `float` | 0.0-1.0 extraction confidence |
| | `.context` | `str` | Relevant quote from the current turn |

**Side effects**:
- Accumulated into `orchestrator.accumulated_allegations`
- Persisted as `AllegationStatement` evidence nodes via the Tool Gateway
- Entity keys used to resolve `missing_fields`

---

### 2. Auth Agent (conditional)

**Role**: Assess impersonation risk of the caller.

**Condition**: Runs if `cm_turn <= 3` OR `impersonation_risk >= 0.4`.

| Direction | Field | Type | Description |
|-----------|-------|------|-------------|
| **INPUT** | `transcript_text` | `str` | Current turn's raw text (behavioral cues) |
| **INPUT** | `auth_events` | `list[dict]` | Auth event records from retrieval (failed attempts, device fingerprints, login history) |
| **INPUT** | `customer_profile` | `dict \| None` | Customer profile from retrieval (call patterns, geo, recent account changes) |
| **INPUT** | `conversation_history` | `list[(speaker, text)] \| None` | Recent conversation turns for multi-turn behavioral pattern detection |
| **INPUT** | `model_provider` | `ModelProvider` | LLM provider for inference |

| Direction | Field | Type | Description |
|-----------|-------|------|-------------|
| **OUTPUT** | `impersonation_risk` | `float` | 0.0-1.0 — LOW (0.0-0.3), MED (0.3-0.6), HIGH (0.6-0.8), CRITICAL (0.8+) |
| **OUTPUT** | `risk_factors` | `list[str]` | e.g. "Hesitation on account details", "Device fingerprint mismatch" |
| **OUTPUT** | `step_up_recommended` | `bool` | Whether step-up authentication is recommended |
| **OUTPUT** | `step_up_method` | `str` | `NONE` \| `SMS_OTP` \| `CALLBACK` \| `SECURITY_QUESTIONS` |
| **OUTPUT** | `assessment_summary` | `str` | Brief explanation of the overall assessment |

**Side effects**:
- Updates `orchestrator.impersonation_risk`
- Appends `risk_factors` and step-up flag to `risk_flags`

---

### 3. Retrieval Agent (idempotent)

**Role**: Fetch all case data via Tool Gateway. Runs once per session, returns cached result after.

| Direction | Field | Type | Description |
|-----------|-------|------|-------------|
| **INPUT** | `case_id` | `str` | Case identifier |
| **INPUT** | `call_id` | `str` | Current call identifier |
| **INPUT** | `gateway` | `ToolGateway` | Mediated data access |
| **INPUT** | `model_provider` | `ModelProvider` | LLM provider for inference |

**Tools used** (via `CopilotContext`):
- `tool_lookup_transactions` — TRANSACTION-type evidence nodes (PAN-masked)
- `tool_query_auth_logs` — AUTH_EVENT-type evidence nodes
- `tool_fetch_customer_profile` — CUSTOMER-type evidence node

| Direction | Field | Type | Description |
|-----------|-------|------|-------------|
| **OUTPUT** | `transactions` | `list[dict]` | Transaction records (PAN-masked) |
| **OUTPUT** | `auth_events` | `list[dict]` | Authentication event records |
| **OUTPUT** | `customer_profile` | `dict \| None` | Customer profile (if found) |
| **OUTPUT** | `retrieval_summary` | `str` | Plain-language summary of what was found |
| **OUTPUT** | `data_gaps` | `list[str]` | e.g. "No auth events for disputed txn period" |

**Side effects**:
- Cached in `orchestrator._retrieval_result`
- Fed into Auth Agent (`auth_events`, `customer_profile`) and Hypothesis Agent (`evidence_summary`)

---

## Phase 2 — Hypothesis + Case Advisor (parallel on turn 4+)

### 4. Hypothesis Agent

**Role**: Score 4 investigation categories as a probability distribution.

| Direction | Field | Type | Description |
|-----------|-------|------|-------------|
| **INPUT** | `allegations_summary` | `str` | All accumulated allegations formatted with types, descriptions, confidence, and entities |
| **INPUT** | `auth_summary` | `str` | Formatted auth assessment (impersonation risk, risk factors, step-up method, summary) |
| **INPUT** | `evidence_summary` | `str` | Structured JSON of transactions, auth events, customer profile from retrieval |
| **INPUT** | `current_scores` | `dict[str, float]` | Previous turn's hypothesis scores (Bayesian prior): `{THIRD_PARTY_FRAUD, FIRST_PARTY_FRAUD, SCAM, DISPUTE}` |
| **INPUT** | `conversation_summary` | `str` | Last 5 turns + total turn count |
| **INPUT** | `model_provider` | `ModelProvider` | LLM provider for inference |

| Direction | Field | Type | Description |
|-----------|-------|------|-------------|
| **OUTPUT** | `scores` | `dict[str, float]` | `{THIRD_PARTY_FRAUD: 0.XX, FIRST_PARTY_FRAUD: 0.XX, SCAM: 0.XX, DISPUTE: 0.XX}` (sums to ~1.0) |
| **OUTPUT** | `reasoning` | `dict[str, str]` | Per-category explanation (1-3 sentences each) |
| **OUTPUT** | `contradictions` | `list[str]` | Detected contradictions between allegations and evidence |
| **OUTPUT** | `assessment_summary` | `str` | 2-4 sentence overall assessment |

**Side effects**:
- Updates `orchestrator.hypothesis_scores`

---

### 5. Case Advisor (parallel with Hypothesis on turn 4+)

**Role**: Policy-aware case eligibility assessment with integrated question planning and stopping condition.

**Condition**: Skipped on turns 1-3 (not enough info yet). On turn 4+, runs in parallel with Hypothesis Agent using the **previous turn's** hypothesis scores (acceptable since scores shift incrementally via the Bayesian prior design).

| Direction | Field | Type | Description |
|-----------|-------|------|-------------|
| **INPUT** | `allegations_summary` | `str` | All accumulated allegations (same as hypothesis) |
| **INPUT** | `evidence_summary` | `str` | Retrieved evidence JSON (same as hypothesis) |
| **INPUT** | `hypothesis_scores` | `dict[str, float]` | Previous turn's scores (Bayesian prior) |
| **INPUT** | `conversation_window` | `list[(speaker, text)]` | Assessment-based conversation window (context + new turns) |
| **INPUT** | `missing_fields` | `list[str] \| None` | Entity-based missing fields from triage extraction |
| **INPUT** | `recent_questions` | `list[str] \| None` | Previously suggested questions (last 3 turns, flattened) for dedup |
| **INPUT** | `model_provider` | `ModelProvider` | LLM provider for inference |

**Context**: Embedded policy documents from `docs/policies/*.md` (loaded at module import time).

| Direction | Field | Type | Description |
|-----------|-------|------|-------------|
| **OUTPUT** | `assessments` | `list[CaseTypeAssessment]` | One per case type (fraud, dispute, scam) |
| | `.case_type` | `str` | `"fraud"`, `"dispute"`, or `"scam"` |
| | `.eligibility` | `str` | `"eligible"` \| `"blocked"` \| `"incomplete"` |
| | `.met_criteria` | `list[str]` | Criteria that are satisfied |
| | `.unmet_criteria` | `list[str]` | Criteria not yet satisfied |
| | `.blockers` | `list[str]` | Active blocking rules with explanations |
| | `.policy_citations` | `list[str]` | Specific policy text cited |
| **OUTPUT** | `general_warnings` | `list[str]` | Cross-cutting warnings (escalation triggers, etc.) |
| **OUTPUT** | `questions` | `list[str]` | 0-3 suggested next-best questions (empty when sufficient) |
| **OUTPUT** | `rationale` | `list[str]` | Parallel list — why each question matters |
| **OUTPUT** | `priority_field` | `str` | Most important missing field targeted, or "" when sufficient |
| **OUTPUT** | `information_sufficient` | `bool` | `True` when all required info is gathered per policy checklist |
| **OUTPUT** | `summary` | `str` | 2-4 sentence eligibility landscape + next steps |

**Stopping condition**: When the leading hypothesis case type is `eligible` or all types are `blocked` for non-resolvable reasons, `information_sufficient` is set to `True` and `questions` is empty. This signals to the CCP that enough information has been gathered to proceed.

**Side effects**:
- Questions tracked in `orchestrator._recent_suggestions` (rolling window of last 3 turns for dedup)

---

## Final Output

All agent results are assembled into a single `CopilotSuggestion`:

| Field | Type | Source |
|-------|------|--------|
| `call_id` | `str` | TranscriptEvent |
| `timestamp_ms` | `int` | TranscriptEvent |
| `suggested_questions` | `list[str]` | Case Advisor |
| `risk_flags` | `list[str]` | Auth Agent + all agent error flags |
| `retrieved_facts` | `list[str]` | Retrieval Agent summary |
| `running_summary` | `str` | Accumulated allegations summary |
| `safety_guidance` | `str` | Orchestrator logic (impersonation risk + missing fields) |
| `hypothesis_scores` | `dict[str, float]` | Hypothesis Agent |
| `impersonation_risk` | `float` | Auth Agent |
| `case_eligibility` | `list[dict]` | Case Advisor assessments |
| `case_advisory_summary` | `str` | Case Advisor summary |
| `information_sufficient` | `bool` | Case Advisor stopping signal |

---

## Data Flow Summary

```
Triage --> accumulated_allegations --> Hypothesis --> hypothesis_scores
                                   |                                  |
                                   +-> Case Advisor <-----------------+
                                   |   (eligibility + questions       |
                                   |    + stopping signal)            |
Retrieval --> transactions     ----+                                  |
           |  auth_events ---------+                                  |
           +- customer_profile ----+                                  |
                  |                                                   |
                  +-> Auth Agent --> impersonation_risk ---------------+
                                 |  risk_flags
                                 +-> Hypothesis (auth_summary)
```

---

## Key Design Points

- **Hub-and-spoke**: The orchestrator explicitly controls which agents run and when. No free handoffs.
- **5 specialist agents**: Triage, Auth, Retrieval, Hypothesis, Case Advisor. The former Question Planner has been merged into Case Advisor for reduced latency and policy-aware question generation.
- **Conditional auth**: Auth agent is skipped after turn 3 if impersonation risk drops below 0.4, saving an LLM call.
- **Retrieval with cache invalidation**: Retrieval caches its result but invalidates when triage persists new allegations (evidence store changed). Re-fetches in parallel on the next assessment turn.
- **Case advisor gating + parallelism**: Skipped on turns 1-3. On turn 4+, runs in parallel with Hypothesis Agent using the previous turn's scores.
- **Integrated question planning**: Case Advisor generates 0-3 questions when info is incomplete and returns `information_sufficient=True` with 0 questions when ready.
- **Question dedup**: Case Advisor receives the last 3 turns of suggested questions to avoid repetition.
- **All agents traced**: Every invocation is logged to the trace store with agent name, duration, and status.
- **Error isolation**: Each agent is wrapped in a `_run_*_safe` method. Failures append to `risk_flags` but never crash the pipeline.

---

## Observability (LangFuse)

Optional LangFuse integration provides full LLM observability on top of the built-in SQLite trace store.

### Setup

Set three environment variables (in `.env` or shell):

```bash
# Option 1: LangFuse Cloud (for testing)
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...

# Option 2: Self-hosted (for enterprise)
LANGFUSE_BASE_URL=http://localhost:3000
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

When `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set, LangFuse is enabled automatically at app startup. When unset, the system operates normally without LangFuse.

### What's Captured

**Auto-instrumented** (via `openinference-instrumentation-openai-agents`):
- Every `Runner.run()` LLM call across all 5 agents: prompts, completions, tokens, model, latency
- Tool invocations with arguments and return values
- Agent handoffs

**Orchestrator-added context** (via `propagate_attributes` + `start_as_current_observation`):
- `session_id` — groups all turns for one case
- Phase spans: `phase1_parallel`, `phase2_hypothesis_advisor`
- Turn metadata: `cm_turn`, `assess_interval`

### Trace Hierarchy

```
Trace: "copilot_turn" (session_id=case_id)
├─ Span: "phase1_parallel"
│  ├─ auto: triage → LLM generation + tool calls
│  ├─ auto: auth → LLM generation
│  └─ auto: retrieval → LLM generation + tool calls
└─ Span: "phase2_hypothesis_advisor"
   ├─ auto: hypothesis → LLM generation
   └─ auto: case_advisor → LLM generation
```

### Self-hosted Deployment

```bash
git clone https://github.com/langfuse/langfuse.git
cd langfuse
docker compose up -d
```

Then set `LANGFUSE_BASE_URL=http://localhost:3000` and create API keys in the LangFuse UI under Settings → API Keys.
