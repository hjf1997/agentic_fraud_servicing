"""AMEX-branded Gradio dashboard for simulation results.

Read-only dashboard that displays simulation results from SQLite databases.
No LLM credentials required — all data is pre-computed by the simulation runner.

Entry point: ``python -m agentic_fraud_servicing.ui.dashboard``
"""

import json
import os
import tempfile

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
from pyvis.network import Network

from agentic_fraud_servicing.ui.dashboard_data import (
    discover_scenarios,
    load_audit_trail,
    load_case,
    load_case_pack,
    load_copilot_final_state,
    load_copilot_suggestions,
    load_evidence,
    load_transcript_turns,
)

matplotlib.use("Agg")

# -- AMEX color palette -----------------------------------------------------------

AMEX_BLUE = "#006FCF"
AMEX_NAVY = "#00175A"
AMEX_LIGHT_BLUE = "#B3E0FF"
AMEX_BG = "#F7F8FA"
AMEX_WHITE = "#FFFFFF"

# -- CSS --------------------------------------------------------------------------

DASHBOARD_CSS = """
/* Global overrides */
.gradio-container { font-family: 'Helvetica Neue', Arial, sans-serif; }

/* Section header cards */
.section-header {
    background: linear-gradient(135deg, %(navy)s 0%%, %(blue)s 100%%);
    color: %(white)s; padding: 14px 24px; border-radius: 8px 8px 0 0;
    font-size: 1.25em; font-weight: 700; margin-top: 16px;
    letter-spacing: 0.5px;
}

/* Generic card wrapper */
.card {
    background: %(white)s; border: 1px solid #E0E4EA;
    border-radius: 8px; padding: 20px; margin-bottom: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}

/* Status badges */
.badge { display: inline-block; padding: 3px 10px; border-radius: 12px;
         font-size: 0.85em; font-weight: 600; }
.badge-open   { background: #FFF3CD; color: #856404; }
.badge-investigating { background: #CCE5FF; color: #004085; }
.badge-done   { background: #D4EDDA; color: #155724; }
.badge-closed { background: #D1ECF1; color: #0C5460; }

/* Category badges */
.cat-badge { display: inline-block; padding: 3px 10px; border-radius: 12px;
             font-size: 0.85em; font-weight: 600; }
.cat-third-party { background: #CCE5FF; color: #004085; }
.cat-first-party { background: #F8D7DA; color: #721C24; }
.cat-scam        { background: #FFF3CD; color: #856404; }
.cat-dispute     { background: #D4EDDA; color: #155724; }

/* Chat bubbles */
.chat-container { max-height: 600px; overflow-y: auto; padding: 10px;
                  background: %(bg)s; border-radius: 8px; }
.bubble { padding: 10px 14px; border-radius: 14px; margin: 6px 0;
          max-width: 85%%; line-height: 1.5; font-size: 0.95em; }
.bubble-label { font-size: 0.75em; color: #666; margin-bottom: 2px; }
.bubble-ccp { background: %(blue)s; color: #fff; margin-left: auto;
              text-align: left; border-bottom-right-radius: 4px; }
.bubble-cm  { background: #E8EAED; color: #222; margin-right: auto;
              border-bottom-left-radius: 4px; }
.bubble-sys { background: %(light_blue)s; color: %(navy)s;
              margin: 6px auto; text-align: center; font-style: italic;
              border-radius: 8px; max-width: 70%%; }

/* Evidence table row colors */
.fact-row td { background: #E8F5E9 !important; }
.allegation-row td { background: #FFF3E0 !important; }

/* Copilot turn accordion */
.copilot-detail { background: %(bg)s; padding: 10px; border-radius: 6px;
                  font-size: 0.9em; }
.copilot-detail dt { font-weight: 600; color: %(navy)s; margin-top: 8px; }
.copilot-detail dd { margin-left: 12px; margin-bottom: 4px; }
""" % {
    "navy": AMEX_NAVY,
    "blue": AMEX_BLUE,
    "light_blue": AMEX_LIGHT_BLUE,
    "bg": AMEX_BG,
    "white": AMEX_WHITE,
}

BASE_DIR = "data/simulation"


# -- HTML builders ----------------------------------------------------------------


def _status_badge(status: str) -> str:
    """Render a case status as a colored badge."""
    css_class = {
        "OPEN": "badge-open",
        "INVESTIGATING": "badge-investigating",
        "PENDING_REVIEW": "badge-investigating",
        "CLOSED": "badge-closed",
    }.get(status.upper() if status else "", "badge-open")
    return f'<span class="badge {css_class}">{status}</span>'


def _category_badge(category: str) -> str:
    """Render an investigation category as a colored badge."""
    css_map = {
        "THIRD_PARTY_FRAUD": "cat-third-party",
        "FIRST_PARTY_FRAUD": "cat-first-party",
        "SCAM": "cat-scam",
        "DISPUTE": "cat-dispute",
    }
    css_class = css_map.get(category.upper() if category else "", "")
    return f'<span class="cat-badge {css_class}">{category}</span>'


def _build_case_overview_html(case: dict | None) -> str:
    """Build HTML for the Case Overview section."""
    if not case:
        return '<div class="card"><p>No case data available.</p></div>'

    status = case.get("status", "UNKNOWN")
    allegation = case.get("allegation_type", "N/A")
    txns = case.get("transactions_in_scope", [])

    txn_rows = ""
    for t in txns:
        txn_rows += (
            f"<tr><td>{t.get('transaction_id', 'N/A')}</td>"
            f"<td>${t.get('amount', 0):,.2f}</td>"
            f"<td>{t.get('merchant_name', 'N/A')}</td>"
            f"<td>{t.get('transaction_date', 'N/A')}</td></tr>"
        )

    txn_table = ""
    if txn_rows:
        txn_table = f"""
        <h4 style="margin-top:14px;">Transactions in Scope</h4>
        <table style="width:100%; border-collapse:collapse; font-size:0.9em;">
          <thead>
            <tr style="background:{AMEX_BLUE}; color:#fff;">
              <th style="padding:6px 10px; text-align:left;">Transaction ID</th>
              <th style="padding:6px 10px; text-align:left;">Amount</th>
              <th style="padding:6px 10px; text-align:left;">Merchant</th>
              <th style="padding:6px 10px; text-align:left;">Date</th>
            </tr>
          </thead>
          <tbody>{txn_rows}</tbody>
        </table>"""

    return f"""<div class="card">
      <table style="width:100%; font-size:0.95em;">
        <tr><td style="width:140px; font-weight:600; color:{AMEX_NAVY};">Case ID</td>
            <td>{case.get("case_id", "N/A")}</td>
            <td style="width:140px; font-weight:600; color:{AMEX_NAVY};">Status</td>
            <td>{_status_badge(status)}</td></tr>
        <tr><td style="font-weight:600; color:{AMEX_NAVY};">Allegation</td>
            <td>{allegation}</td>
            <td style="font-weight:600; color:{AMEX_NAVY};">Customer ID</td>
            <td>{case.get("customer_id", "N/A")}</td></tr>
        <tr><td style="font-weight:600; color:{AMEX_NAVY};">Account ID</td>
            <td>{case.get("account_id", "N/A")}</td>
            <td style="font-weight:600; color:{AMEX_NAVY};">Created</td>
            <td>{case.get("created_at", "N/A")}</td></tr>
      </table>
      {txn_table}
    </div>"""


def _build_transcript_html(turns: list[dict]) -> str:
    """Build chat-bubble HTML for the conversation transcript."""
    if not turns:
        return '<div class="card"><p>No transcript data available.</p></div>'

    bubbles = ""
    for t in turns:
        speaker = t.get("speaker", "").upper()
        text = t.get("text", "")
        turn_num = t.get("turn", "")

        if speaker == "CCP":
            label = (
                f'<div class="bubble-label" style="text-align:right;">Turn {turn_num} — CCP</div>'
            )
            bubbles += (
                f'<div style="display:flex; flex-direction:column;'
                f' align-items:flex-end;">{label}'
                f'<div class="bubble bubble-ccp">{text}</div></div>'
            )
        elif speaker in ("CARDMEMBER", "CM"):
            label = f'<div class="bubble-label">Turn {turn_num} — Cardmember</div>'
            bubbles += (
                f'<div style="display:flex; flex-direction:column;'
                f' align-items:flex-start;">{label}'
                f'<div class="bubble bubble-cm">{text}</div></div>'
            )
        else:
            label = (
                f'<div class="bubble-label" style="text-align:center;">'
                f"Turn {turn_num} — {speaker}</div>"
            )
            bubbles += (
                f'<div style="display:flex; flex-direction:column;'
                f' align-items:center;">{label}'
                f'<div class="bubble bubble-sys">{text}</div></div>'
            )

    return f'<div class="chat-container">{bubbles}</div>'


def _build_hypothesis_chart(suggestions: list[dict]) -> plt.Figure | None:
    """Build a matplotlib line chart of hypothesis score evolution."""
    if not suggestions:
        return None

    turns: list[int] = []
    scores: dict[str, list[float]] = {
        "THIRD_PARTY_FRAUD": [],
        "FIRST_PARTY_FRAUD": [],
        "SCAM": [],
        "DISPUTE": [],
    }

    for s in suggestions:
        turn = s.get("turn", 0)
        suggestion = s.get("suggestion", {})
        h_scores = suggestion.get("hypothesis_scores", {})
        turns.append(turn)
        for key in scores:
            scores[key].append(h_scores.get(key, 0.0))

    if not turns:
        return None

    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = {
        "THIRD_PARTY_FRAUD": AMEX_BLUE,
        "FIRST_PARTY_FRAUD": "#D32F2F",
        "SCAM": "#F57C00",
        "DISPUTE": "#388E3C",
    }
    for key, vals in scores.items():
        ax.plot(
            turns,
            vals,
            marker="o",
            markersize=4,
            linewidth=2,
            color=colors[key],
            label=key.replace("_", " ").title(),
        )

    ax.set_xlabel("Turn", fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Hypothesis Score Evolution", fontsize=12, fontweight="bold", color=AMEX_NAVY)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _build_copilot_final_html(state: dict | None) -> str:
    """Build HTML for the final copilot state card."""
    if not state:
        return '<div class="card"><p>No copilot final state available.</p></div>'

    h_scores = state.get("hypothesis_scores", {})
    score_items = "".join(
        f"<li><strong>{k.replace('_', ' ').title()}</strong>: {v:.2f}</li>"
        for k, v in h_scores.items()
    )
    imp_risk = state.get("impersonation_risk", 0.0)
    missing = state.get("missing_fields", [])
    missing_str = ", ".join(missing) if missing else "None"

    return f"""<div class="card">
      <h4 style="color:{AMEX_NAVY}; margin-top:0;">Final Copilot State</h4>
      <p><strong>Impersonation Risk:</strong> {imp_risk:.2f}</p>
      <p><strong>Hypothesis Scores:</strong></p>
      <ul style="margin:4px 0;">{score_items}</ul>
      <p><strong>Missing Fields:</strong> {missing_str}</p>
    </div>"""


def _build_copilot_turns_html(suggestions: list[dict]) -> str:
    """Build accordion HTML for per-turn copilot details."""
    if not suggestions:
        return '<div class="card"><p>No copilot suggestions available.</p></div>'

    items = ""
    for s in suggestions:
        turn = s.get("turn", "?")
        sug = s.get("suggestion", {})
        questions = sug.get("suggested_questions", [])
        risk_flags = sug.get("risk_flags", [])
        summary = sug.get("running_summary", "")
        safety = sug.get("safety_guidance", "")

        q_list = "".join(f"<li>{q}</li>" for q in questions) if questions else "<li>None</li>"
        r_list = "".join(f"<li>{r}</li>" for r in risk_flags) if risk_flags else "<li>None</li>"

        items += f"""
        <details style="margin-bottom:6px;">
          <summary style="cursor:pointer; font-weight:600; color:{AMEX_BLUE};
                          padding:6px;">Turn {turn}</summary>
          <div class="copilot-detail">
            <dl>
              <dt>Suggested Questions</dt><dd><ul>{q_list}</ul></dd>
              <dt>Risk Flags</dt><dd><ul>{r_list}</ul></dd>
              <dt>Running Summary</dt><dd>{summary or "N/A"}</dd>
              <dt>Safety Guidance</dt><dd>{safety or "N/A"}</dd>
            </dl>
          </div>
        </details>"""

    return f'<div class="card">{items}</div>'


def _build_evidence_graph_interactive(nodes: list[dict], edges: list[dict]) -> str:
    """Build an interactive network graph of evidence nodes and edges using pyvis.

    Returns an HTML string containing a self-contained vis.js network graph
    that can be embedded directly in a Gradio gr.HTML component.
    """
    if not nodes and not edges:
        return '<div class="card"><p>No evidence data available for graph.</p></div>'

    net = Network(
        height="550px",
        width="100%",
        directed=True,
        bgcolor=AMEX_WHITE,
        font_color=AMEX_NAVY,
    )

    # Physics settings for a readable layout
    net.set_options(
        json.dumps(
            {
                "physics": {
                    "enabled": True,
                    "solver": "forceAtlas2Based",
                    "forceAtlas2Based": {
                        "gravitationalConstant": -80,
                        "centralGravity": 0.01,
                        "springLength": 180,
                        "springConstant": 0.04,
                        "damping": 0.5,
                    },
                    "stabilization": {"iterations": 150},
                },
                "interaction": {
                    "hover": True,
                    "tooltipDelay": 100,
                    "navigationButtons": True,
                    "keyboard": {"enabled": True},
                },
                "edges": {
                    "arrows": {"to": {"enabled": True, "scaleFactor": 0.8}},
                    "smooth": {"type": "curvedCW", "roundness": 0.2},
                    "font": {"size": 10, "color": "#666666", "strokeWidth": 0},
                },
                "nodes": {
                    "font": {"size": 12, "face": "Helvetica Neue, Arial, sans-serif"},
                    "borderWidth": 2,
                    "shadow": {"enabled": True, "size": 4, "x": 2, "y": 2},
                },
            }
        )
    )

    # Node styling by type and source
    _NODE_COLORS = {
        "TRANSACTION": {"background": "#E3F2FD", "border": AMEX_BLUE},
        "AUTH_EVENT": {"background": "#F3E5F5", "border": "#7B1FA2"},
        "CARD": {"background": "#E8EAF6", "border": "#283593"},
        "DEVICE": {"background": "#E0F2F1", "border": "#00695C"},
        "CUSTOMER": {"background": "#FFF3E0", "border": "#E65100"},
        "MERCHANT": {"background": "#F1F8E9", "border": "#33691E"},
        "DELIVERY_PROOF": {"background": "#E0F7FA", "border": "#006064"},
        "REFUND_RECORD": {"background": "#FCE4EC", "border": "#880E4F"},
        "CLAIM_STATEMENT": {"background": "#FFF8E1", "border": "#F9A825"},
        "INVESTIGATOR_NOTE": {"background": "#EFEBE9", "border": "#4E342E"},
    }

    _NODE_SHAPES = {
        "TRANSACTION": "dot",
        "AUTH_EVENT": "diamond",
        "CARD": "square",
        "DEVICE": "triangle",
        "CUSTOMER": "star",
        "MERCHANT": "dot",
        "DELIVERY_PROOF": "square",
        "REFUND_RECORD": "dot",
        "CLAIM_STATEMENT": "triangle",
        "INVESTIGATOR_NOTE": "square",
    }

    _EDGE_COLORS = {
        "SUPPORTS": "#388E3C",
        "CONTRADICTS": "#D32F2F",
        "DERIVED_FROM": "#F57C00",
        "FACT": "#1565C0",
        "ALLEGATION": "#FF8F00",
    }

    # Build a node_id set for edge validation
    node_ids = set()

    for n in nodes:
        node_id = n.get("node_id", "")
        if not node_id:
            continue
        node_ids.add(node_id)

        node_type = n.get("node_type", "UNKNOWN")
        source_type = n.get("source_type", "")
        summary = _evidence_node_summary(n)

        colors = _NODE_COLORS.get(node_type, {"background": "#E0E0E0", "border": "#616161"})
        shape = _NODE_SHAPES.get(node_type, "dot")

        # Allegation nodes get an orange border override
        if source_type == "ALLEGATION":
            colors = {**colors, "border": "#FF8F00"}

        # Short label for the graph (truncate long IDs)
        short_id = node_id
        if len(short_id) > 18:
            short_id = short_id[:15] + "..."

        # Tooltip with full details
        tooltip = f"<b>{node_type}</b><br/>ID: {node_id}<br/>Source: {source_type}<br/>{summary}"

        # Size based on node importance
        size = 25 if node_type in ("TRANSACTION", "CLAIM_STATEMENT", "CUSTOMER") else 18

        net.add_node(
            node_id,
            label=short_id,
            title=tooltip,
            shape=shape,
            size=size,
            color=colors,
            borderWidth=3 if source_type == "ALLEGATION" else 2,
        )

    for e in edges:
        source = e.get("source_node_id", "")
        target = e.get("target_node_id", "")
        edge_type = e.get("edge_type", "")

        # Skip edges referencing nodes not in the graph
        if source not in node_ids or target not in node_ids:
            continue

        color = _EDGE_COLORS.get(edge_type, "#999999")
        width = 3 if edge_type in ("CONTRADICTS", "SUPPORTS") else 2

        net.add_edge(
            source,
            target,
            title=edge_type,
            label=edge_type.replace("_", " ").title(),
            color=color,
            width=width,
            dashes=edge_type == "DERIVED_FROM",
        )

    # Generate HTML via pyvis, then embed in an iframe so vis.js scripts execute
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        tmp_path = f.name
    try:
        net.save_graph(tmp_path)
        with open(tmp_path) as f:
            full_html = f.read()
    finally:
        os.unlink(tmp_path)

    # Escape the HTML for safe embedding in an iframe srcdoc attribute.
    # srcdoc requires &, <, > and " to be escaped.
    iframe_html = full_html.replace("&", "&amp;").replace('"', "&quot;")

    # Wrap in a styled container with a legend
    legend_html = f"""
    <div style="display:flex; flex-wrap:wrap; gap:12px; padding:10px 0;
                font-size:0.85em; color:{AMEX_NAVY};">
      <span><b>Node Shapes:</b></span>
      <span>&#9679; Transaction/Merchant</span>
      <span>&#9670; Auth Event</span>
      <span>&#9733; Customer</span>
      <span>&#9650; Device/Claim</span>
      <span>&#9632; Card/Delivery/Note</span>
      <span style="margin-left:16px;"><b>Edges:</b></span>
      <span style="color:#388E3C;">&#9644; Supports</span>
      <span style="color:#D32F2F;">&#9644; Contradicts</span>
      <span style="color:#F57C00;">- - Derived From</span>
    </div>
    <div style="display:flex; gap:12px; padding:0 0 10px 0;
                font-size:0.85em; color:{AMEX_NAVY};">
      <span><b>Source:</b></span>
      <span style="border:2px solid #283593; padding:1px 8px;
                   border-radius:4px; background:#E8EAF6;">FACT</span>
      <span style="border:2px solid #FF8F00; padding:1px 8px;
                   border-radius:4px; background:#FFF8E1;">ALLEGATION</span>
      <span style="margin-left:12px; color:#666;">
        Drag nodes to rearrange. Scroll to zoom. Hover for details.</span>
    </div>"""

    return f"""<div class="card">
        {legend_html}
        <div style="border:1px solid #E0E4EA; border-radius:8px; overflow:hidden;">
            <iframe srcdoc="{iframe_html}"
                    style="width:100%; height:570px; border:none;"
                    sandbox="allow-scripts allow-same-origin">
            </iframe>
        </div>
    </div>"""


def _build_evidence_html(nodes: list[dict], edges: list[dict]) -> str:
    """Build HTML tables for evidence nodes and edges."""
    if not nodes and not edges:
        return '<div class="card"><p>No evidence data available.</p></div>'

    # Nodes table
    node_rows = ""
    for n in nodes:
        source = n.get("source_type", "")
        row_class = "fact-row" if source == "FACT" else "allegation-row"
        node_type = n.get("node_type", "N/A")
        node_id = n.get("node_id", "N/A")
        # Build a short summary from key fields
        summary = _evidence_node_summary(n)
        node_rows += (
            f'<tr class="{row_class}">'
            f"<td style='padding:6px 10px;'>{node_id[:20]}</td>"
            f"<td style='padding:6px 10px;'>{node_type}</td>"
            f"<td style='padding:6px 10px;'>{source}</td>"
            f"<td style='padding:6px 10px;'>{summary}</td></tr>"
        )

    nodes_html = ""
    if node_rows:
        nodes_html = f"""
        <h4 style="color:{AMEX_NAVY};">Evidence Nodes</h4>
        <table style="width:100%; border-collapse:collapse; font-size:0.9em;">
          <thead>
            <tr style="background:{AMEX_BLUE}; color:#fff;">
              <th style="padding:6px 10px; text-align:left;">Node ID</th>
              <th style="padding:6px 10px; text-align:left;">Type</th>
              <th style="padding:6px 10px; text-align:left;">Source</th>
              <th style="padding:6px 10px; text-align:left;">Summary</th>
            </tr>
          </thead>
          <tbody>{node_rows}</tbody>
        </table>"""

    # Edges table
    edge_rows = ""
    for e in edges:
        edge_rows += (
            f"<tr><td style='padding:6px 10px;'>{e.get('source_node_id', 'N/A')[:20]}</td>"
            f"<td style='padding:6px 10px;'>{e.get('target_node_id', 'N/A')[:20]}</td>"
            f"<td style='padding:6px 10px;'>{e.get('edge_type', 'N/A')}</td></tr>"
        )

    edges_html = ""
    if edge_rows:
        edges_html = f"""
        <h4 style="color:{AMEX_NAVY}; margin-top:16px;">Evidence Edges</h4>
        <table style="width:100%; border-collapse:collapse; font-size:0.9em;">
          <thead>
            <tr style="background:{AMEX_BLUE}; color:#fff;">
              <th style="padding:6px 10px; text-align:left;">Source Node</th>
              <th style="padding:6px 10px; text-align:left;">Target Node</th>
              <th style="padding:6px 10px; text-align:left;">Edge Type</th>
            </tr>
          </thead>
          <tbody>{edge_rows}</tbody>
        </table>"""

    return f'<div class="card">{nodes_html}{edges_html}</div>'


def _evidence_node_summary(node: dict) -> str:
    """Extract a short summary string from an evidence node dict."""
    node_type = node.get("node_type", "")
    if node_type == "TRANSACTION":
        amt = node.get("amount", "")
        merchant = node.get("merchant_name", "")
        return f"${amt} at {merchant}" if amt else merchant or "Transaction"
    if node_type == "AUTH_EVENT":
        return f"{node.get('auth_type', '')} — {node.get('result', '')}"
    if node_type == "CLAIM_STATEMENT":
        claim_type = node.get("claim_type", "")
        text = node.get("text", "")
        prefix = f"[{claim_type}] " if claim_type else ""
        full = f"{prefix}{text}"
        return full[:100] + "..." if len(full) > 100 else full
    if node_type == "INVESTIGATOR_NOTE":
        text = node.get("text", "")
        return text[:80] + "..." if len(text) > 80 else text
    if node_type == "CUSTOMER":
        return node.get("customer_id", "") or "Customer record"
    if node_type == "MERCHANT":
        return node.get("merchant_id", "") or node.get("category", "") or "Merchant"
    if node_type == "DELIVERY_PROOF":
        return f"{node.get('status', '')} — {node.get('tracking_id', '')}"
    if node_type == "CARD":
        return f"Status: {node.get('status', 'N/A')}"
    if node_type == "DEVICE":
        return f"Enrolled: {node.get('enrolment_date', 'N/A')}"
    return json.dumps(node, default=str)[:60]


def _build_investigation_html(case_pack: dict | None) -> str:
    """Build HTML for the Investigation Results section."""
    if not case_pack:
        return '<div class="card"><p>Investigation not yet completed.</p></div>'

    # Case summary narrative
    summary = case_pack.get("case_summary", "")
    summary_paragraphs = "".join(f"<p>{p.strip()}</p>" for p in summary.split("\n\n") if p.strip())

    # Decision recommendation card
    rec = case_pack.get("decision_recommendation", {})
    category = rec.get("category", "UNKNOWN")
    confidence = rec.get("confidence", 0.0)
    confidence_pct = int(confidence * 100)

    # Confidence bar color
    bar_color = AMEX_BLUE if confidence >= 0.7 else "#F57C00" if confidence >= 0.4 else "#D32F2F"

    # Top factors
    factors_html = ""
    for f in rec.get("top_factors", []):
        weight = f.get("weight", 0)
        weight_pct = int(weight * 100)
        factors_html += (
            f'<li style="margin-bottom:8px;">'
            f'<span style="font-weight:600;">{f.get("factor", "")}</span>'
            f'<br/><span style="font-size:0.85em; color:#666;">'
            f"Evidence: {f.get('evidence_ref', 'N/A')} · "
            f'Weight: <span style="background:{AMEX_BLUE}; color:#fff; '
            f'padding:1px 6px; border-radius:8px; font-size:0.85em;">'
            f"{weight_pct}%</span></span></li>"
        )

    # Uncertainties
    uncert_html = "".join(f"<li>{u}</li>" for u in rec.get("uncertainties", []))

    # Suggested actions
    actions_html = "".join(f"<li>{a}</li>" for a in rec.get("suggested_actions", []))

    # Required approvals
    approvals = rec.get("required_approvals", [])
    approvals_html = "".join(
        f'<span class="badge badge-investigating" style="margin-right:6px;">'
        f"{a.replace('_', ' ').title()}</span>"
        for a in approvals
    )

    # Timeline
    timeline = case_pack.get("timeline", [])
    timeline_rows = ""
    for entry in timeline:
        source = entry.get("source", "")
        src_class = "fact-row" if source == "FACT" else "allegation-row"
        timeline_rows += (
            f'<tr class="{src_class}">'
            f'<td style="padding:6px 10px; white-space:nowrap;">'
            f"{entry.get('timestamp', 'N/A')[:19]}</td>"
            f'<td style="padding:6px 10px;">{entry.get("event_type", "N/A")}</td>'
            f'<td style="padding:6px 10px;">{entry.get("description", "")}</td>'
            f'<td style="padding:6px 10px;">{source}</td></tr>'
        )

    timeline_html = ""
    if timeline_rows:
        timeline_html = f"""
        <h4 style="color:{AMEX_NAVY}; margin-top:20px;">
          Investigation Timeline ({len(timeline)} events)</h4>
        <table style="width:100%; border-collapse:collapse; font-size:0.9em;">
          <thead>
            <tr style="background:{AMEX_BLUE}; color:#fff;">
              <th style="padding:6px 10px; text-align:left;">Timestamp</th>
              <th style="padding:6px 10px; text-align:left;">Event Type</th>
              <th style="padding:6px 10px; text-align:left;">Description</th>
              <th style="padding:6px 10px; text-align:left;">Source</th>
            </tr>
          </thead>
          <tbody>{timeline_rows}</tbody>
        </table>"""

    # Investigation notes
    notes = case_pack.get("investigation_notes", [])
    notes_html = ""
    if notes:
        notes_items = "".join(f"<li style='margin-bottom:6px;'>{n}</li>" for n in notes)
        notes_html = f"""
        <h4 style="color:{AMEX_NAVY}; margin-top:20px;">Investigation Notes</h4>
        <ul style="padding-left:20px;">{notes_items}</ul>"""

    return f"""<div class="card">
      <h4 style="color:{AMEX_NAVY}; margin-top:0;">Case Summary</h4>
      <div style="line-height:1.7; color:#333;">{summary_paragraphs}</div>

      <div style="margin-top:20px; padding:20px; background:{AMEX_BG};
                  border-radius:8px; border-left:4px solid {AMEX_BLUE};">
        <h4 style="color:{AMEX_NAVY}; margin-top:0;">Decision Recommendation</h4>
        <div style="display:flex; align-items:center; gap:16px; margin-bottom:16px;">
          <div>{_category_badge(category)}</div>
          <div style="flex:1;">
            <div style="font-size:0.85em; color:#666; margin-bottom:4px;">
              Confidence: {confidence_pct}%</div>
            <div style="background:#E0E4EA; border-radius:6px; height:16px;
                        overflow:hidden; display:grid;
                        grid-template-columns:{confidence_pct}fr {100 - confidence_pct}fr;">
              <div style="background:{bar_color}; border-radius:6px;"></div>
              <div></div>
            </div>
          </div>
        </div>

        <h5 style="color:{AMEX_NAVY}; margin:12px 0 6px;">Top Factors</h5>
        <ol style="padding-left:20px;">{factors_html}</ol>

        <h5 style="color:{AMEX_NAVY}; margin:12px 0 6px;">Uncertainties</h5>
        <ul style="padding-left:20px; color:#856404;">{uncert_html}</ul>

        <h5 style="color:{AMEX_NAVY}; margin:12px 0 6px;">Suggested Actions</h5>
        <ol style="padding-left:20px;">{actions_html}</ol>

        <h5 style="color:{AMEX_NAVY}; margin:12px 0 6px;">Required Approvals</h5>
        <div>{approvals_html if approvals_html else "None"}</div>
      </div>

      {timeline_html}
      {notes_html}
    </div>"""


def _build_audit_trail_html(traces: list[dict]) -> str:
    """Build HTML for the Audit Trail section."""
    if not traces:
        return '<div class="card"><p>No audit trail data available.</p></div>'

    rows = ""
    for t in traces:
        agent = t.get("agent_id", "N/A")
        action = t.get("action", "N/A")
        duration = t.get("duration_ms", 0.0)
        status = t.get("status", "success")
        timestamp = t.get("timestamp", "N/A")
        if isinstance(timestamp, str) and len(timestamp) > 19:
            timestamp = timestamp[:19]

        status_style = "color:#155724;" if status == "success" else "color:#721C24;"
        rows += (
            f"<tr>"
            f"<td style='padding:5px 10px; white-space:nowrap;'>{timestamp}</td>"
            f"<td style='padding:5px 10px;'>{agent}</td>"
            f"<td style='padding:5px 10px;'>{action}</td>"
            f"<td style='padding:5px 10px; text-align:right;'>"
            f"{duration:.0f}ms</td>"
            f"<td style='padding:5px 10px; {status_style}'>{status}</td></tr>"
        )

    return f"""<div class="card">
      <table style="width:100%; border-collapse:collapse; font-size:0.85em;">
        <thead>
          <tr style="background:{AMEX_BLUE}; color:#fff;">
            <th style="padding:5px 10px; text-align:left;">Timestamp</th>
            <th style="padding:5px 10px; text-align:left;">Agent</th>
            <th style="padding:5px 10px; text-align:left;">Action</th>
            <th style="padding:5px 10px; text-align:right;">Duration</th>
            <th style="padding:5px 10px; text-align:left;">Status</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </div>"""


# -- Main load callback -----------------------------------------------------------


def _load_scenario(scenario_name: str) -> tuple:
    """Load all data for a scenario and return component updates.

    Returns a tuple of 9 items matching the output components.
    """
    if not scenario_name:
        empty = '<div class="card"><p>Select a scenario and click Load.</p></div>'
        return empty, empty, None, empty, empty, empty, empty, empty, empty

    db_dir = os.path.join(BASE_DIR, scenario_name)

    # Load case
    case = load_case(db_dir)
    case_id = case.get("case_id", "") if case else ""

    # Load all data
    turns = load_transcript_turns(db_dir, case_id) if case_id else []
    suggestions = load_copilot_suggestions(db_dir, case_id) if case_id else []
    final_state = load_copilot_final_state(db_dir, case_id) if case_id else None
    nodes, edges = load_evidence(db_dir, case_id) if case_id else ([], [])
    case_pack = load_case_pack(db_dir, case_id) if case_id else None
    traces = load_audit_trail(db_dir, case_id) if case_id else []

    # Build HTML / charts
    return (
        _build_case_overview_html(case),
        _build_transcript_html(turns),
        _build_hypothesis_chart(suggestions),
        _build_copilot_final_html(final_state),
        _build_copilot_turns_html(suggestions),
        _build_evidence_graph_interactive(nodes, edges),
        _build_evidence_html(nodes, edges),
        _build_investigation_html(case_pack),
        _build_audit_trail_html(traces),
    )


# -- App factory ------------------------------------------------------------------


def create_dashboard_app() -> gr.Blocks:
    """Create the AMEX-branded simulation results dashboard.

    Returns:
        A configured gr.Blocks instance ready to launch.
    """
    scenarios = discover_scenarios(BASE_DIR)

    with gr.Blocks(title="AMEX Fraud Simulation Dashboard") as app:
        # Inject CSS via <style> tag (Gradio 6 moved css param to launch)
        gr.HTML(f"<style>{DASHBOARD_CSS}</style>")

        # -- Top bar ---------------------------------------------------------------
        gr.HTML(
            f"""<div style="background: linear-gradient(135deg, {AMEX_NAVY}, {AMEX_BLUE});
                 padding: 16px 24px; border-radius: 8px; margin-bottom: 16px;">
              <h1 style="color:#fff; margin:0; font-size:1.5em;">
                AMEX Fraud Simulation Dashboard</h1>
              <p style="color:{AMEX_LIGHT_BLUE}; margin:4px 0 0 0; font-size:0.9em;">
                Read-only view of simulation results</p>
            </div>"""
        )

        with gr.Row():
            scenario_dropdown = gr.Dropdown(
                choices=scenarios,
                label="Scenario",
                scale=3,
            )
            load_btn = gr.Button("Load Scenario", variant="primary", scale=1)

        # -- Sections ---------------------------------------------------------------
        # Section 1: Case Overview
        gr.HTML('<div class="section-header" style="color:white;">Case Overview</div>')
        case_html = gr.HTML()

        # Section 2 & 3: Copilot Analysis + Conversation
        gr.HTML(
            '<div class="section-header" style="color:white;">'
            "Conversation &amp; Copilot Analysis</div>"
        )
        # Top row: hypothesis chart (left) + final copilot state (right)
        with gr.Row():
            with gr.Column(scale=3):
                chart_plot = gr.Plot(label="Hypothesis Scores")
            with gr.Column(scale=2):
                final_state_html = gr.HTML()
        # Bottom row: transcript (left) + per-turn copilot details (right)
        with gr.Row():
            with gr.Column(scale=1):
                transcript_html = gr.HTML()
            with gr.Column(scale=1):
                copilot_turns_html = gr.HTML()

        # Section 4: Evidence Graph
        gr.HTML('<div class="section-header" style="color:white;">Evidence Graph</div>')
        evidence_graph_html = gr.HTML()
        with gr.Accordion("Evidence Tables (Detail View)", open=False):
            evidence_html = gr.HTML()

        # Section 5: Investigation Results
        gr.HTML('<div class="section-header" style="color:white;">Investigation Results</div>')
        investigation_html = gr.HTML()

        # Section 6: Audit Trail (collapsible)
        with gr.Accordion("Audit Trail", open=False):
            audit_trail_html = gr.HTML()

        # -- Wire load callback ----------------------------------------------------
        load_btn.click(
            fn=_load_scenario,
            inputs=[scenario_dropdown],
            outputs=[
                case_html,
                transcript_html,
                chart_plot,
                final_state_html,
                copilot_turns_html,
                evidence_graph_html,
                evidence_html,
                investigation_html,
                audit_trail_html,
            ],
        )

    return app


def main() -> None:
    """Launch the dashboard."""
    app = create_dashboard_app()
    app.launch()


if __name__ == "__main__":
    main()
