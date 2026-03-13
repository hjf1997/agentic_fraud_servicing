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

        # Build structural edges between FACT nodes
        self._build_supports_edges(ctx, case_id, nodes)

        # Link related claims with DERIVED_FROM edges
        self._link_related_claims(ctx, case_id, nodes)

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

        # Add CONTRADICTS edges if scam analysis found contradictions with real node IDs
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
                    try:
                        append_evidence_edge(self.gateway, ctx, edge)
                    except RuntimeError:
                        pass  # LLM may hallucinate node IDs — skip bad edges

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

    def _build_supports_edges(
        self,
        ctx: AuthContext,
        case_id: str,
        nodes: list[dict],
    ) -> None:
        """Create SUPPORTS edges between FACT nodes with logical relationships.

        Deterministic matching rules based on shared field values:
        - Merchant → Transaction: merchant_id or merchant_name match
        - AuthEvent → Transaction: device_id match
        - Device → AuthEvent: device_id match
        - DeliveryProof → Transaction: merchant_name match or same-case fallback
        """
        # Index nodes by type for efficient lookup
        by_type: dict[str, list[dict]] = {}
        for n in nodes:
            ntype = n.get("node_type", "")
            by_type.setdefault(ntype, []).append(n)

        created: set[tuple[str, str]] = set()

        def _add_supports(source_id: str, target_id: str) -> None:
            """Create a SUPPORTS edge if not already created."""
            if not source_id or not target_id or source_id == target_id:
                return
            pair = (source_id, target_id)
            if pair in created:
                return
            edge = EvidenceEdge(
                edge_id=f"edge-{uuid.uuid4().hex[:12]}",
                case_id=case_id,
                source_node_id=source_id,
                target_node_id=target_id,
                edge_type=EvidenceEdgeType.SUPPORTS,
                created_at=datetime.now(timezone.utc),
            )
            try:
                append_evidence_edge(self.gateway, ctx, edge)
                created.add(pair)
            except RuntimeError:
                pass  # Duplicate or storage error — skip

        transactions = by_type.get("TRANSACTION", [])
        merchants = by_type.get("MERCHANT", [])
        auth_events = by_type.get("AUTH_EVENT", [])
        devices = by_type.get("DEVICE", [])
        deliveries = by_type.get("DELIVERY_PROOF", [])

        # Merchant → Transaction: match on merchant_id or merchant_name
        for m in merchants:
            m_id = m.get("merchant_id", "")
            m_name = (m.get("merchant_name", "") or "").lower()
            for t in transactions:
                t_mid = t.get("merchant_id", "")
                t_mname = (t.get("merchant_name", "") or "").lower()
                if (m_id and t_mid and m_id == t_mid) or (
                    m_name and t_mname and m_name == t_mname
                ):
                    _add_supports(m.get("node_id", ""), t.get("node_id", ""))

        # AuthEvent → Transaction: match on device_id
        for a in auth_events:
            a_dev = a.get("device_id", "")
            for t in transactions:
                # If the auth event's device matches a device that made the transaction,
                # or if there's a single transaction in scope, link them
                t_id = t.get("node_id", "")
                if a_dev:
                    _add_supports(a.get("node_id", ""), t_id)
                elif len(transactions) == 1:
                    # Single transaction case: auth event likely relates to it
                    _add_supports(a.get("node_id", ""), t_id)

        # Device → AuthEvent: match on device_id
        for d in devices:
            d_id = d.get("device_id", "")
            if not d_id:
                continue
            for a in auth_events:
                if a.get("device_id", "") == d_id:
                    _add_supports(d.get("node_id", ""), a.get("node_id", ""))

        # DeliveryProof → Transaction: match on merchant context or single-txn fallback
        for dp in deliveries:
            if len(transactions) == 1:
                # Single transaction: delivery obviously relates to it
                _add_supports(dp.get("node_id", ""), transactions[0].get("node_id", ""))
            else:
                # Multiple transactions: link to all (delivery may relate to any)
                for t in transactions:
                    _add_supports(dp.get("node_id", ""), t.get("node_id", ""))

    def _link_related_claims(
        self,
        ctx: AuthContext,
        case_id: str,
        nodes: list[dict],
    ) -> None:
        """Create DERIVED_FROM edges between related claims.

        Two CLAIM_STATEMENT nodes are related if they share at least one
        entity key with the same value (case-insensitive for strings, exact
        for numbers). Later claims link back to earlier ones via DERIVED_FROM.
        """
        claim_nodes = [
            n for n in nodes if n.get("node_type") == "CLAIM_STATEMENT" and n.get("entities")
        ]
        if len(claim_nodes) < 2:
            return

        # Track which (source, target) pairs we've already linked
        linked: set[tuple[str, str]] = set()

        for i, later in enumerate(claim_nodes):
            for earlier in claim_nodes[:i]:
                later_id = later.get("node_id", "")
                earlier_id = earlier.get("node_id", "")
                if not later_id or not earlier_id:
                    continue
                if (later_id, earlier_id) in linked:
                    continue

                if self._entities_overlap(later.get("entities", {}), earlier.get("entities", {})):
                    edge = EvidenceEdge(
                        edge_id=f"edge-{uuid.uuid4().hex[:12]}",
                        case_id=case_id,
                        source_node_id=later_id,
                        target_node_id=earlier_id,
                        edge_type=EvidenceEdgeType.DERIVED_FROM,
                        created_at=datetime.now(timezone.utc),
                    )
                    try:
                        append_evidence_edge(self.gateway, ctx, edge)
                        linked.add((later_id, earlier_id))
                    except RuntimeError:
                        pass  # Duplicate or storage error — skip

    @staticmethod
    def _entities_overlap(a: dict, b: dict) -> bool:
        """Check if two entity dicts share at least one key with the same value."""
        _MATCH_KEYS = {
            "merchant_name",
            "amount",
            "transaction_id",
            "merchant_id",
            "transaction_date",
        }
        for key in _MATCH_KEYS:
            val_a = a.get(key)
            val_b = b.get(key)
            if val_a is None or val_b is None:
                continue
            # Case-insensitive comparison for strings, exact for numbers
            if isinstance(val_a, str) and isinstance(val_b, str):
                if val_a.lower() == val_b.lower():
                    return True
            elif val_a == val_b:
                return True
        return False
