"""Central post-call investigator orchestrator with gateway integration.

Reads case and evidence from the gateway, runs all specialist agents
sequentially (scheme mapper, merchant evidence, scam detector, case writer),
writes results back to the evidence graph, and produces the final CasePack.
This is a plain Python class — not an Agents SDK Agent — keeping the
control flow explicit and auditable.
"""

import json
import uuid
from datetime import datetime, timezone

from agents import ModelProvider

from agentic_fraud_servicing.gateway.tool_gateway import AuthContext, ToolGateway
from agentic_fraud_servicing.gateway.tools.write_tools import (
    append_evidence_edge,
    append_evidence_node,
    update_case_status,
)
from agentic_fraud_servicing.investigator.case_writer import CasePack, run_case_writer
from agentic_fraud_servicing.investigator.merchant_evidence import (
    MerchantAnalysis,
    run_merchant_analysis,
)
from agentic_fraud_servicing.investigator.scam_detector import (
    ScamAnalysis,
    run_scam_detection,
)
from agentic_fraud_servicing.investigator.scheme_mapper import (
    SchemeMappingResult,
    run_scheme_mapping,
)
from agentic_fraud_servicing.models.enums import (
    CaseStatus,
    EvidenceEdgeType,
    EvidenceSourceType,
)
from agentic_fraud_servicing.models.evidence import EvidenceEdge, InvestigatorNote


class InvestigatorOrchestrator:
    """Central orchestrator for the post-call investigation pipeline.

    Reads case data and evidence via the gateway, runs all specialist agents
    sequentially, writes investigation results back, and returns the final
    CasePack. Uses a hub-and-spoke pattern with explicit control flow.

    Attributes:
        gateway: ToolGateway for mediated data access.
        model_provider: LLM provider for all specialist agents.
    """

    def __init__(self, gateway: ToolGateway, model_provider: ModelProvider) -> None:
        self.gateway = gateway
        self.model_provider = model_provider

    async def investigate(self, case_id: str) -> CasePack:
        """Run the full post-call investigation pipeline for a case.

        Reads case + evidence, runs all 4 specialist agents, writes results
        back to the evidence store, updates case status, and returns the
        CasePack. Each specialist call is wrapped in try/except so failures
        are graceful — the orchestrator continues with remaining specialists.

        Args:
            case_id: The case identifier to investigate.

        Returns:
            CasePack from the case writer agent with investigation results.

        Raises:
            RuntimeError: If the case is not found or the case writer fails.
        """
        ctx = AuthContext(
            agent_id="investigator",
            case_id=case_id,
            permissions={"read", "write"},
        )
        errors: list[str] = []

        # 1. Read case and evidence from gateway
        case = self.gateway.case_store.get_case(case_id)
        if case is None:
            raise RuntimeError(f"Case not found: {case_id}")
        self.gateway.log_call(
            ctx=ctx,
            action="get_case",
            input_summary=f'{{"case_id": "{case_id}"}}',
            output_summary='{"found": true}',
            duration_ms=0.0,
            status="success",
        )

        nodes = self.gateway.evidence_store.get_nodes_by_case(case_id)
        edges = self.gateway.evidence_store.get_edges_by_case(case_id)
        self.gateway.log_call(
            ctx=ctx,
            action="get_evidence",
            input_summary=f'{{"case_id": "{case_id}"}}',
            output_summary=f'{{"nodes": {len(nodes)}, "edges": {len(edges)}}}',
            duration_ms=0.0,
            status="success",
        )

        # Separate nodes by source_type and node_type
        claims = [n for n in nodes if n.get("source_type") == "ALLEGATION"]
        facts = [n for n in nodes if n.get("source_type") == "FACT"]
        merchant_nodes = [n for n in nodes if n.get("node_type") == "MERCHANT"]
        transaction_nodes = [n for n in nodes if n.get("node_type") == "TRANSACTION"]

        # 2. Build case summary text
        allegation = case.allegation_type.value if case.allegation_type else "unknown"
        case_summary = (
            f"Case {case_id}: {allegation} allegation. "
            f"Customer {case.customer_id}, Account {case.account_id}. "
            f"Evidence: {len(nodes)} nodes, {len(edges)} edges. "
            f"Claims: {len(claims)}, Facts: {len(facts)}."
        )

        # Extract claim texts for specialists
        claim_texts = [c.get("text", "") for c in claims if c.get("text")]
        evidence_summary = json.dumps(facts[:10], indent=2, default=str)

        # 3. Run scheme mapper
        scheme_result = await self._run_scheme_mapping_safe(
            case_summary,
            allegation,
            claim_texts,
            evidence_summary,
            errors,
        )

        # 4. Run merchant evidence analysis
        merchant_result = await self._run_merchant_analysis_safe(
            merchant_nodes,
            transaction_nodes,
            errors,
        )

        # 5. Run scam detection
        transcript_summary = ""
        if case.timeline:
            transcript_summary = "; ".join(e.description for e in case.timeline[:20])
        scam_result = await self._run_scam_detection_safe(
            claims,
            facts,
            transcript_summary,
            errors,
        )

        # 6. Run case writer — assembles the final CasePack
        case_data_str = case.model_dump_json()
        scheme_str = scheme_result.model_dump_json() if scheme_result else "{}"
        merchant_str = merchant_result.model_dump_json() if merchant_result else "{}"
        scam_str = scam_result.model_dump_json() if scam_result else "{}"

        case_pack = await run_case_writer(
            case_data=case_data_str,
            scheme_result=scheme_str,
            merchant_result=merchant_str,
            scam_result=scam_str,
            model_provider=self.model_provider,
        )

        # 7. Write results back via gateway
        # Add InvestigatorNote with case summary
        note = InvestigatorNote(
            node_id=f"note-{uuid.uuid4().hex[:12]}",
            case_id=case_id,
            source_type=EvidenceSourceType.FACT,
            created_at=datetime.now(timezone.utc),
            text=case_pack.case_summary,
            author="investigator",
        )
        append_evidence_node(self.gateway, ctx, note)

        # Add CONTRADICTS edges if scam analysis found contradictions
        if scam_result and scam_result.contradictions:
            for contradiction in scam_result.contradictions:
                source_id = contradiction.get("claim_node_id", "")
                target_id = contradiction.get("evidence_node_id", "")
                if source_id and target_id:
                    edge = EvidenceEdge(
                        edge_id=f"edge-{uuid.uuid4().hex[:12]}",
                        case_id=case_id,
                        source_node_id=source_id,
                        target_node_id=target_id,
                        edge_type=EvidenceEdgeType.CONTRADICTS,
                        created_at=datetime.now(timezone.utc),
                    )
                    append_evidence_edge(self.gateway, ctx, edge)

        # Update case status to INVESTIGATING
        update_case_status(self.gateway, ctx, case_id, CaseStatus.INVESTIGATING)

        return case_pack

    # -- Private helper methods --

    async def _run_scheme_mapping_safe(
        self,
        case_summary: str,
        allegation_type: str,
        claims: list[str],
        evidence_summary: str,
        errors: list[str],
    ) -> SchemeMappingResult | None:
        """Run scheme mapper with error handling."""
        try:
            return await run_scheme_mapping(
                case_summary=case_summary,
                allegation_type=allegation_type,
                claims=claims,
                evidence_summary=evidence_summary,
                model_provider=self.model_provider,
            )
        except Exception as exc:
            errors.append(f"Scheme mapping failed: {exc}")
            return None

    async def _run_merchant_analysis_safe(
        self,
        merchant_nodes: list[dict],
        transaction_nodes: list[dict],
        errors: list[str],
    ) -> MerchantAnalysis | None:
        """Run merchant analysis with error handling."""
        try:
            return await run_merchant_analysis(
                merchant_evidence=merchant_nodes,
                transaction_evidence=transaction_nodes,
                model_provider=self.model_provider,
            )
        except Exception as exc:
            errors.append(f"Merchant analysis failed: {exc}")
            return None

    async def _run_scam_detection_safe(
        self,
        claims: list[dict],
        facts: list[dict],
        transcript_summary: str,
        errors: list[str],
    ) -> ScamAnalysis | None:
        """Run scam detection with error handling."""
        try:
            return await run_scam_detection(
                claims=claims,
                facts=facts,
                transcript_summary=transcript_summary,
                model_provider=self.model_provider,
            )
        except Exception as exc:
            errors.append(f"Scam detection failed: {exc}")
            return None
