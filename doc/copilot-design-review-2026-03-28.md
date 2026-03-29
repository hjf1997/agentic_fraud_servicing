# Copilot Design Review — 2026-03-28

Comprehensive review of the agentic fraud servicing copilot system covering
agent architecture, fraud servicing domain, regulatory compliance, latency,
observability, and code quality. Each item is grounded in the actual codebase.

---

## 1. Agent Architecture

### [P1] Pre-Computed Analytical Signals for Hypothesis Agent

**Issue**: The hypothesis agent receives raw transaction data via
`_format_evidence_for_hypothesis()` (orchestrator.py:484-523), which serializes
full transaction dicts as JSON. While the agent sees all transactions, it relies
on the LLM to infer patterns like velocity anomalies, geo-impossibility, or
spending deviation from baseline. An experienced fraud analyst works with
pre-computed metrics, not raw records.

**Impact**: Fraud detection accuracy and consistency. LLMs may miss subtle
patterns in raw data that computed signals would surface reliably (e.g.,
"3 transactions in 10 minutes across 3 countries" or "spending 400% above
90-day average"). Pre-computed signals also reduce the LLM's cognitive load,
improving consistency.

**Implementation effort**: M (1-3 days). Add a preprocessing step before
hypothesis agent invocation that computes: transaction velocity
(count/timewindow), geographic spread, spending deviation from historical
baseline. Feed these as structured metrics alongside the raw evidence.

**Latency consideration**: Neutral. Computation is deterministic and fast.
Reduces LLM token consumption by supplementing raw data with concise signals.

---

### [P1] Add Merchant Risk Profiling Agent

**Issue**: No agent profiles the merchant involved in the dispute. The
`investigator/merchant_evidence.py` does merchant analysis post-call, but
during the live copilot session, merchant-level signals (chargeback history,
merchant category risk tier, known fraud-prone patterns) are unavailable.
The hypothesis agent treats all merchants equivalently.

**Impact**: Fraud detection accuracy. Subscription disputes at high-chargeback
merchants should be weighted differently than one-time purchases at reputable
retailers. Merchant context is a standard signal in fraud investigation.

**Implementation effort**: M (1-3 days). New gateway tool
(`get_merchant_profile`) + either a new agent or integration into the retrieval
agent's tool set. Merchant data feeds into hypothesis scoring.

**Latency consideration**: Near-zero if added to Phase 1 `asyncio.gather()`.
Only adds latency if the merchant data lookup is slower than existing Phase 1
agents.

---

### [P1] Hypothesis Agent: Dispute-Type-Specific Reasoning

**Issue**: `hypothesis_agent.py` uses a single set of 10 reasoning patterns for
all dispute types (lines 87-128). Subscription fraud (RECURRING_AFTER_CANCEL)
has fundamentally different indicators than CNP fraud (CARD_NOT_PRESENT_FRAUD)
or goods disputes (GOODS_NOT_RECEIVED). The scoring rubric doesn't specialize
based on the triage-extracted `AllegationDetailType`.

**Impact**: Fraud detection accuracy. Generic patterns work well for common
cases but miss nuances for specific dispute types (e.g., subscription disputes
need cancellation evidence analysis, not chip+PIN authentication checks).

**Implementation effort**: M (1-3 days). Add conditional reasoning pattern sets
selected based on triage output's primary `AllegationDetailType`. The hypothesis
agent already receives allegations summary -- it just needs type-aware reasoning
guidance.

**Latency consideration**: Neutral. Same LLM call, different prompt content.

---

### [P2] Reduce Phase 2 Sequential Dependency

**Issue**: Question planner (Phase 2b) waits for both hypothesis and
case_advisor (Phase 2a) to complete. It consumes `unmet_criteria` from the case
advisor (orchestrator.py:324-327, passed via `planner_missing` at line 341).
However, the question planner's primary inputs are hypothesis scores and missing
fields -- it could start with those and incorporate case advisor unmet criteria
as supplementary input.

**Impact**: CCP experience (latency). The question planner could start as soon
as hypothesis scores are available, without waiting for case_advisor completion.

**Implementation effort**: S (< 1 day). Use `asyncio.create_task` for question
planner triggered by hypothesis completion, rather than waiting for the full
Phase 2a gather.

**Latency consideration**: Reduces latency by overlapping question planning with
case advisor completion.

---

### [P2] Consider Merging Case Advisor and Question Planner

**Issue**: After turn 3, case_advisor runs in parallel with hypothesis
(orchestrator.py:303), then question_planner runs sequentially (line 342). Both
produce CCP-facing guidance -- case_advisor produces eligibility assessments and
"next info needed," while question_planner produces investigative questions.
They share overlapping input: allegations, evidence, hypothesis scores,
conversation context.

**Impact**: Latency (saves one sequential LLM call). Simpler pipeline. Risk:
combined prompt may be too complex for a single agent to handle well.

**Implementation effort**: M (1-3 days). Merge prompts, combine output schemas.
The policy document loading from case_advisor adds complexity to the merge.

**Latency consideration**: Reduces latency by eliminating one sequential LLM
call on the critical path.

---

## 2. Fraud Servicing Domain

### [P1] No Dispute History Analysis

**Issue**: The system analyzes each dispute in isolation. The gateway tools
(`read_tools.py`) provide `lookup_transactions`, `query_auth_logs`, and
`fetch_customer_profile` -- but no dispute history tool. There's no way to check
whether the cardholder has filed previous disputes, their dispute frequency, or
outcomes. The `FIRST_PARTY_FRAUD` reasoning patterns in hypothesis_agent.py
mention "pattern of similar disputes" (enums.py line 103) but the system has no
data to support this check.

**Impact**: Fraud detection accuracy. Repeat disputers and serial first-party
fraud patterns are a strong signal. A cardholder filing their 10th dispute in 3
months should be treated very differently than a first-time disputer.

**Implementation effort**: M (1-3 days). Add `get_dispute_history` gateway tool,
include in retrieval agent's toolset, feed output into hypothesis agent.

**Latency consideration**: Adds a data fetch to Phase 1. Runs in parallel with
existing retrieval tools, so net impact depends on backend response time.

---

### [P1] Friendly Fraud Detection -- Missing Behavioral Pattern Analysis

**Issue**: The hypothesis agent's reasoning patterns cover some first-party fraud
indicators (chip+PIN contradiction, delivery proof, merchant familiarity).
However, there's no mechanism to detect behavioral patterns across the
conversation that indicate friendly fraud: the cardholder's dispute history (see
above), patterns of claiming "unauthorized" on transactions that match their
normal spending profile, or progressive story changes. The scam_detector in the
investigator pipeline has a `contradiction_level` axis but runs post-call only.

**Impact**: Fraud detection accuracy. First-party fraud / friendly fraud is
explicitly called out as a cross-cutting category in the system's own taxonomy
(INVESTIGATION_CATEGORIES_REFERENCE, enums.py:88-109), yet the real-time copilot
has limited tools to detect it beyond what the LLM infers from individual turn
contradictions.

**Implementation effort**: M (1-3 days). Enhance hypothesis agent prompts with
behavioral consistency tracking across turns. Feed dispute history data when
available.

**Latency consideration**: Neutral. Prompt enhancement in existing LLM call.

---

### [P1] ATO-Specific Investigation Workflow Gaps

**Issue**: The triage agent correctly extracts `ACCOUNT_TAKEOVER` allegations
(triage_agent.py:65-68), and the auth agent assesses impersonation risk.
However, the downstream pipeline doesn't differentiate ATO from other fraud
types. The retrieval agent has no tool for account change history (password
resets, email changes, phone number modifications), login session history, or
device enrollment timeline. The hypothesis agent's reasoning patterns mention
"device fingerprint mismatches" but the evidence to support this check is
limited to what `query_auth_logs` returns.

**Impact**: Case resolution quality. ATO cases need credential compromise
timeline, account modification history, and multi-session analysis -- data
sources the current retrieval agent can't access.

**Implementation effort**: L (1-2 weeks). New gateway tools for account change
history and login sessions. Enhance retrieval agent to call these when triage
identifies ATO-related allegations. Update hypothesis patterns for ATO-specific
evidence evaluation.

**Latency consideration**: Adds latency for ATO cases only (additional data
fetches). Should be conditional on triage output.

---

### [P2] Limited Cardmember Experience Support

**Issue**: The `question_planner.py` generates investigative questions and the
`case_advisor.py` provides policy-based eligibility guidance. Neither produces
empathy cues, de-escalation language, or next-step communication guidance for
the CCP. The CopilotSuggestion model has `safety_guidance`
(orchestrator.py:552-560) but it only covers impersonation warnings and missing
fields -- not emotional support or communication strategy.

**Impact**: Customer experience. Fraud disputes are stressful. CCPs need
guidance on how to communicate, not just what to investigate.

**Implementation effort**: S (< 1 day). Add a communication guidance section to
the case_advisor prompt. No new agent needed.

**Latency consideration**: Neutral. Additional prompt content in existing LLM
call.

---

## 3. Regulatory & Compliance

### [P0] No Reg E / Reg Z Timeline Tracking

**Issue**: No agent or tool calculates or surfaces regulatory deadlines. Reg E
requires provisional credit within 10 business days for debit disputes and
investigation completion within 45 days (90 days for certain cases). Reg Z has
different timelines for credit card billing errors. The existing `compliance.py`
provides `check_retention` (7-year window), `verify_consent`, and
`redact_case_fields` -- but no regulatory deadline enforcement. The case
advisor's policy documents (3 files: `fraud_case_checklist.md`,
`dispute_case_checklist.md`, `general_guidelines.md`) may reference timelines in
text, but there's no deterministic deadline calculation or alerting.

**Impact**: Regulatory risk -- material. Missing Reg E/Z deadlines exposes AMEX
to regulatory penalties and CFPB enforcement actions.

**Implementation effort**: M (1-3 days). Add `check_regulatory_deadlines` to
`compliance.py`. Deterministic calculation based on dispute type and filing
date -- no LLM needed. Surface deadlines in case_advisor output or as a
standalone risk flag.

**Latency consideration**: Neutral. Deterministic computation, no LLM call.

---

### [P0] No BSA/AML Escalation Path

**Issue**: When fraud patterns suggest money laundering or structuring, SAR
filing obligations override normal dispute handling. The system has no AML
indicator detection and no escalation workflow. The hypothesis agent and
investigator pipeline assess fraud categories but don't flag patterns that
trigger BSA/AML obligations (e.g., structuring: transactions just below
reporting thresholds).

**Impact**: Regulatory risk -- severe. Failure to identify and escalate
SAR-worthy patterns is a federal compliance issue.

**Implementation effort**: M (1-3 days). Add AML pattern recognition to
hypothesis agent prompts or as a post-hypothesis check. When triggered, surface
an immediate escalation flag to the CCP. This is flagging/routing, not a full
AML system.

**Latency consideration**: Neutral to slight addition. Can be integrated into
existing hypothesis analysis.

---

### [P1] No Audit Trail for Model Reasoning

**Issue**: The `TraceStore` (trace_store.py) records agent name, duration, input
`{"turn": N}`, and output `{"status": "success"}` -- but not the actual
reasoning or decision data. The `input_data` logged in `_log_agent_trace`
(orchestrator.py:572) is just `json.dumps({"turn": self._turn_count})`, not the
actual agent input. The `output_data` is just `json.dumps({"status": status})`,
not the hypothesis scores or case advisor recommendations. LangFuse now captures
full prompts/completions, but it's an observability tool, not an audit system
with retention guarantees.

**Impact**: Regulatory risk, audit compliance. Regulators need explainable
decision chains for dispute outcomes. The current trace data is insufficient to
answer "why was this classified as likely_fraud?"

**Implementation effort**: L (1-2 weeks). Either enrich `_log_agent_trace` to
capture actual agent inputs/outputs, or build a separate audit store with
structured decision records, model version tracking, and retention policies.

**Latency consideration**: Neutral. Audit logging is async.

---

### [P1] PCI-DSS: PAN Exposure in LLM Prompts

**Issue**: The `tool_gateway.py:mask_pan_in_dict()` masks PANs in tool outputs
(using `_PAN_RE` regex). However, cardholder-spoken PAN data in the transcript
flows through `_format_conversation_for_hypothesis()` (orchestrator.py:525-537)
which passes raw transcript text directly to the LLM. If a cardholder reads
their full card number during the call, it enters the LLM prompt unmasked.
LangFuse traces now capture full prompts (bedrock_provider.py generation_span),
creating another PAN storage location.

**Impact**: PCI-DSS compliance risk. PAN data in LLM prompts is transmitted to
the LLM provider and stored in LangFuse traces.

**Implementation effort**: M (1-3 days). Add PAN regex masking to transcript
text before it enters the agent pipeline (e.g., in the orchestrator before
constructing conversation windows). Configure LangFuse data retention/redaction
policies.

**Latency consideration**: Minimal. Regex replacement is fast.

---

## 4. Latency & Performance

### [P2] Policy Documents Could Be Selectively Loaded (Future Concern)

**Issue**: `case_advisor.py:112` loads all 3 policy files at module import and
embeds them in the system prompt. Currently the corpus is small (3 files:
`fraud_case_checklist.md`, `dispute_case_checklist.md`,
`general_guidelines.md`), so token impact is modest. However, as the policy
corpus grows (more dispute types, more regulations, jurisdiction-specific
policies), prompt size will grow linearly.

**Impact**: Currently minor. Becomes a cost/latency concern if policy corpus
exceeds ~20 pages.

**Implementation effort**: M (1-3 days) when needed. Implement selective loading
based on triage-identified dispute type, or use a tool-based approach where the
case advisor requests specific policies.

**Latency consideration**: Would reduce latency when implemented, by shrinking
prompts.

---

## 5. Observability & Quality

### [P1] No Evaluation Framework for Copilot Quality

**Issue**: There's no systematic way to measure whether the copilot is helping
CCPs. Traces (TraceStore + LangFuse) capture execution data but no quality
metrics. Are hypothesis scores accurate? Are suggested questions useful? Are
case advisor recommendations correct? The evaluation layer (L1-17 through L1-20
in tasks.json) is planned but not yet integrated.

**Impact**: System reliability. Without evaluation, quality degradation from
model updates or data distribution shifts goes undetected.

**Implementation effort**: L (1-2 weeks). Evaluation pipeline with labeled test
cases and scoring rubrics per agent output.

**Latency consideration**: Offline only.

---

### [P2] No Drift Detection or Model Version Tracking

**Issue**: When the underlying LLM is updated, there's no mechanism to detect
quality changes. `BedrockModel` sends model ID to LangFuse traces, but there's
no comparison framework or alerting.

**Impact**: System reliability. Model updates could silently degrade quality.

**Implementation effort**: M (1-3 days). Golden test suite that runs on model
changes with baseline comparisons.

**Latency consideration**: Offline only.

---

## 6. Code Quality & Implementation

### [P1] Error Handling Swallows Failures Without CCP Visibility

**Issue**: All `_run_*_safe` methods catch exceptions and append to `risk_flags`
(e.g., orchestrator.py:619: `risk_flags.append(f"Retrieval failed: {exc}")`).
But `risk_flags` appear in the CopilotSuggestion output alongside normal risk
indicators -- there's no distinction between "impersonation risk detected" and
"hypothesis agent crashed." The CCP has no clear signal that a capability is
degraded.

**Impact**: CCP experience. If the hypothesis agent fails, the CCP gets stale
scores from the previous turn without knowing they're stale. Decisions may be
based on incomplete information without awareness.

**Implementation effort**: S (< 1 day). Add a `degraded_capabilities` field to
CopilotSuggestion that explicitly lists which agents failed this turn, separate
from risk_flags. Surface as a distinct UI indicator.

**Latency consideration**: Neutral.

---

### [P1] Investigator Pipeline Findings Don't Feed Back to Copilot

**Issue**: The `InvestigatorOrchestrator` (investigator/orchestrator.py) runs 4
agents post-call and writes results to the evidence store. But these findings
(merchant risk profiles, scam indicators, scheme mappings) are never surfaced to
the copilot for future calls about similar transactions or merchants. Each live
call starts from zero.

**Impact**: Fraud detection accuracy over time. The investigator discovers
patterns (e.g., "this merchant has a 40% chargeback rate") that could improve
real-time copilot guidance for subsequent cases.

**Implementation effort**: L (1-2 weeks). Design a mechanism where investigator
outputs are stored in a queryable format and the copilot's retrieval agent can
access them for related cases.

**Latency consideration**: Adds a lookup to Phase 1 retrieval. The value of
pre-computed insights likely outweighs the cost.

---

### [P2] Bedrock Provider Streaming Not Implemented

**Issue**: `bedrock_provider.py:stream_response()` raises
`NotImplementedError`. The entire pipeline runs non-streaming -- the CCP sees
nothing until all agents complete for a turn.

**Impact**: CCP experience. During long turns, no progress indication. Streaming
would enable progressive display.

**Implementation effort**: M (1-3 days). Implement using Bedrock's
`converse_stream()` API.

**Latency consideration**: Reduces perceived latency (time-to-first-token
improvement).

---

## Priority Summary

| Priority | Count | Items |
|----------|-------|-------|
| P0 | 2 | Reg E/Z timeline tracking, BSA/AML escalation path |
| P1 | 10 | Pre-computed signals, merchant profiling, dispute-type reasoning, dispute history, friendly fraud detection, ATO workflows, audit trail, PCI-DSS PAN exposure, evaluation framework, error handling visibility, investigator feedback loop |
| P2 | 5 | Phase 2 sequential dependency, merge advisor+planner, cardmember experience, policy selective loading, drift detection, streaming |

### Quick Wins (high value, low effort)

1. **Reg E/Z deadline tracking** (P0, M) -- deterministic, no LLM, high regulatory value
2. **Error handling visibility** (P1, S) -- add `degraded_capabilities` field to CopilotSuggestion
3. **Cardmember experience** (P2, S) -- prompt-only change in case_advisor
4. **Phase 2 sequential dependency** (P2, S) -- overlap question planner with case advisor

---

## Corrections Log

Items from the initial review that were corrected after thorough codebase
verification:

| Original Claim | Correction |
|---|---|
| Hypothesis agent performs single-transaction reasoning | Wrong. `_format_evidence_for_hypothesis()` passes all transactions as structured JSON. Reframed as pre-computed signals gap. |
| Case advisor and question planner run sequentially | Partially wrong. After turn 3, hypothesis + case_advisor run in parallel (`asyncio.gather`). Only question_planner is sequential after them. |
| Phase 2 is fully sequential | Wrong. Hypothesis + case_advisor are already parallelized after turn 3. |
| Policy docs inflate token usage | Overstated. Only 3 small policy files. Future concern only. |
| No caching of gateway responses | Wrong. Orchestrator caches retrieval results with explicit invalidation. Removed from review. |
| AllegationDetailType enum is fragile | Overstated. Deliberate design choice with 17 types covering 100%/95%. Removed from review. |
| Auth runs on every call | Wrong. `_should_run_auth()` already implements conditional execution. Removed from review. |
| No ATO detection | Partially wrong. Triage + auth handle detection. Gap is in investigation workflows and data sources, not detection. |
| compliance.py only handles PCI checks | Wrong. It handles retention, consent, and redaction -- general compliance utilities. |
