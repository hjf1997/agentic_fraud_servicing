"""AMEX-branded Gradio dashboard for simulation results.

Read-only dashboard that displays simulation results from SQLite databases.
No LLM credentials required — all data is pre-computed by the simulation runner.

Entry point: ``python -m agentic_fraud_servicing.ui.dashboard``
"""

import json
import os

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt

from agentic_fraud_servicing.ui.dashboard_data import (
    discover_scenarios,
    load_case,
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
    background: linear-gradient(135deg, %(navy)s, %(blue)s);
    color: #fff; padding: 12px 20px; border-radius: 8px 8px 0 0;
    font-size: 1.1em; font-weight: 600; margin-top: 16px;
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
            f"<td>{t.get('merchant', 'N/A')}</td>"
            f"<td>{t.get('date', 'N/A')}</td></tr>"
        )

    txn_table = ""
    if txn_rows:
        txn_table = f"""
        <h4 style="margin-top:14px;">Transactions in Scope</h4>
        <table style="width:100%%; border-collapse:collapse; font-size:0.9em;">
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
      <table style="width:100%%; font-size:0.95em;">
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
        <table style="width:100%%; border-collapse:collapse; font-size:0.9em;">
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
        <table style="width:100%%; border-collapse:collapse; font-size:0.9em;">
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
        merchant = node.get("merchant", "")
        return f"${amt} at {merchant}" if amt else merchant or "Transaction"
    if node_type == "AUTH_EVENT":
        return f"{node.get('auth_type', '')} — {node.get('result', '')}"
    if node_type == "CLAIM_STATEMENT":
        text = node.get("text", "")
        return text[:80] + "..." if len(text) > 80 else text
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


# -- Main load callback -----------------------------------------------------------


def _load_scenario(scenario_name: str) -> tuple:
    """Load all data for a scenario and return component updates.

    Returns a tuple matching the output components:
        (case_html, transcript_html, chart, final_state_html,
         copilot_turns_html, evidence_html, sections_visible)
    """
    if not scenario_name:
        empty = '<div class="card"><p>Select a scenario and click Load.</p></div>'
        return empty, empty, None, empty, empty, empty, gr.update(visible=False)

    db_dir = os.path.join(BASE_DIR, scenario_name)

    # Load case
    case = load_case(db_dir)
    case_id = case.get("case_id", "") if case else ""

    # Load all data
    turns = load_transcript_turns(db_dir, case_id) if case_id else []
    suggestions = load_copilot_suggestions(db_dir, case_id) if case_id else []
    final_state = load_copilot_final_state(db_dir, case_id) if case_id else None
    nodes, edges = load_evidence(db_dir, case_id) if case_id else ([], [])

    # Build HTML / charts
    case_html = _build_case_overview_html(case)
    transcript_html = _build_transcript_html(turns)
    chart = _build_hypothesis_chart(suggestions)
    final_html = _build_copilot_final_html(final_state)
    turns_html = _build_copilot_turns_html(suggestions)
    evidence_html = _build_evidence_html(nodes, edges)

    return (
        case_html,
        transcript_html,
        chart,
        final_html,
        turns_html,
        evidence_html,
        gr.update(visible=True),
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

        # -- Sections (hidden until loaded) ----------------------------------------
        sections = gr.Column(visible=False)
        with sections:
            # Section 1: Case Overview
            gr.HTML('<div class="section-header">Case Overview</div>')
            case_html = gr.HTML()

            # Section 2 & 3: Transcript (left) + Copilot Analysis (right)
            gr.HTML('<div class="section-header">Conversation &amp; Copilot Analysis</div>')
            with gr.Row():
                with gr.Column(scale=1):
                    transcript_html = gr.HTML()
                with gr.Column(scale=1):
                    chart_plot = gr.Plot(label="Hypothesis Scores")
                    final_state_html = gr.HTML()
                    copilot_turns_html = gr.HTML()

            # Section 4: Evidence Graph
            gr.HTML('<div class="section-header">Evidence Graph</div>')
            evidence_html = gr.HTML()

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
                evidence_html,
                sections,
            ],
        )

    return app


def main() -> None:
    """Launch the dashboard."""
    app = create_dashboard_app()
    app.launch()


if __name__ == "__main__":
    main()
