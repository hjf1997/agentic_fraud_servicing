"""Export simulation results as a self-contained HTML report.

Reads from the SQLite databases under data/simulation/{scenario}/ and produces
a single .html file that can be opened in any browser — no server needed.

Usage:
    python scripts/export_report.py --scenario scam_techvault
    python scripts/export_report.py --scenario scam_techvault -o ~/Desktop/report.html
    python scripts/export_report.py --list
"""

import argparse
import base64
import io
import os
import sys
from datetime import datetime, timezone

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

# Ensure the project root is importable when running as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Reuse the HTML builder helpers and color constants from the dashboard module.
# We import them to avoid duplicating ~700 lines of rendering code.
from agentic_fraud_servicing.ui.dashboard import (  # noqa: E402
    AMEX_BG,
    AMEX_BLUE,
    AMEX_LIGHT_BLUE,
    AMEX_NAVY,
    AMEX_WHITE,
    DASHBOARD_CSS,
    _build_audit_trail_html,
    _build_case_overview_html,
    _build_copilot_final_html,
    _build_copilot_turns_html,
    _build_evidence_graph_interactive,
    _build_evidence_html,
    _build_investigation_html,
    _build_transcript_html,
)
from agentic_fraud_servicing.ui.dashboard_data import (  # noqa: E402
    discover_scenarios,
    load_audit_trail,
    load_case,
    load_case_pack,
    load_copilot_final_state,
    load_copilot_suggestions,
    load_evidence,
    load_transcript_turns,
)

BASE_DIR = "data/simulation"


# ---------------------------------------------------------------------------
# Hypothesis chart → base64 PNG
# ---------------------------------------------------------------------------


def _render_hypothesis_chart_base64(suggestions: list[dict]) -> str:
    """Render the hypothesis score evolution chart as a base64-encoded PNG.

    Returns an <img> tag with the chart embedded, or a placeholder message
    if no suggestion data is available.
    """
    if not suggestions:
        return '<p style="color:#666;">No hypothesis data available.</p>'

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
        return '<p style="color:#666;">No hypothesis data available.</p>'

    colors = {
        "THIRD_PARTY_FRAUD": AMEX_BLUE,
        "FIRST_PARTY_FRAUD": "#D32F2F",
        "SCAM": "#F57C00",
        "DISPUTE": "#388E3C",
    }

    fig, ax = plt.subplots(figsize=(8, 3.5))
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

    # Render to base64 PNG
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")

    return f'<img src="data:image/png;base64,{b64}" style="width:100%; max-width:800px;" />'


def _render_eligibility_chart_base64(suggestions: list[dict]) -> str:
    """Render the case eligibility evolution chart as a base64-encoded PNG.

    Returns an <img> tag with the chart embedded, or a placeholder message
    if no eligibility data is available.
    """
    if not suggestions:
        return '<p style="color:#666;">No eligibility data available.</p>'

    turns: list[int] = []
    case_types_set: set[str] = set()
    elig_data: list[dict[str, str]] = []

    for s in suggestions:
        turn = s.get("turn", 0)
        sug = s.get("suggestion", {})
        ce = sug.get("case_eligibility", [])
        if not ce:
            continue
        turns.append(turn)
        turn_map: dict[str, str] = {}
        for item in ce:
            ct = item.get("case_type", "").lower()
            st = item.get("eligibility", "").lower()
            if ct and st:
                turn_map[ct] = st
                case_types_set.add(ct)
        elig_data.append(turn_map)

    if not turns or not case_types_set:
        return '<p style="color:#666;">No eligibility data available.</p>'

    case_types = sorted(case_types_set)
    status_colors = {"eligible": "#388E3C", "blocked": "#D32F2F", "incomplete": "#F57C00"}

    fig, ax = plt.subplots(figsize=(8, max(1.5, 0.5 * len(case_types) + 0.8)))

    for row_idx, ct in enumerate(case_types):
        for col_idx, turn_map in enumerate(elig_data):
            status = turn_map.get(ct, "")
            color = status_colors.get(status, "#CCCCCC")
            ax.barh(
                row_idx,
                0.8,
                left=col_idx - 0.4,
                height=0.6,
                color=color,
                edgecolor="white",
                linewidth=1,
            )
            if status:
                label = status[0].upper()
                ax.text(
                    col_idx,
                    row_idx,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color="white",
                )

    ax.set_yticks(range(len(case_types)))
    ax.set_yticklabels([ct.title() for ct in case_types], fontsize=9)
    ax.set_xticks(range(len(turns)))
    ax.set_xticklabels([str(t) for t in turns], fontsize=8)
    ax.set_xlabel("Turn", fontsize=10)
    ax.set_title("Case Eligibility by Turn", fontsize=12, fontweight="bold", color=AMEX_NAVY)
    ax.invert_yaxis()

    from matplotlib.patches import Patch

    legend_items = [
        Patch(facecolor="#388E3C", label="Eligible"),
        Patch(facecolor="#D32F2F", label="Blocked"),
        Patch(facecolor="#F57C00", label="Incomplete"),
    ]
    ax.legend(handles=legend_items, fontsize=7, loc="upper right", framealpha=0.8)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")

    return f'<img src="data:image/png;base64,{b64}" style="width:100%; max-width:800px;" />'


# ---------------------------------------------------------------------------
# Evidence graph → inline HTML (pyvis generates self-contained HTML)
# ---------------------------------------------------------------------------


def _render_evidence_graph_inline(nodes: list[dict], edges: list[dict]) -> str:
    """Render the evidence graph as inline HTML.

    The dashboard version uses an iframe with srcdoc. For the static report
    we embed the same content but keep it as a standalone block.
    """
    # Reuse the dashboard builder — it returns HTML with an embedded iframe.
    # This works fine in a standalone HTML file.
    return _build_evidence_graph_interactive(nodes, edges)


# ---------------------------------------------------------------------------
# Full HTML report assembly
# ---------------------------------------------------------------------------


def _build_full_report(scenario_name: str) -> str:
    """Load all data for a scenario and assemble a self-contained HTML report."""
    db_dir = os.path.join(BASE_DIR, scenario_name)

    # Load data
    case = load_case(db_dir)
    case_id = case.get("case_id", "") if case else ""

    turns = load_transcript_turns(db_dir, case_id) if case_id else []
    suggestions = load_copilot_suggestions(db_dir, case_id) if case_id else []
    final_state = load_copilot_final_state(db_dir, case_id) if case_id else None
    nodes, edges = load_evidence(db_dir, case_id) if case_id else ([], [])
    case_pack = load_case_pack(db_dir, case_id) if case_id else None
    traces = load_audit_trail(db_dir, case_id) if case_id else []

    # Build section HTML
    case_overview = _build_case_overview_html(case)
    transcript = _build_transcript_html(turns)
    hypothesis_chart = _render_hypothesis_chart_base64(suggestions)
    eligibility_chart = _render_eligibility_chart_base64(suggestions)
    copilot_final = _build_copilot_final_html(final_state, suggestions)
    copilot_turns = _build_copilot_turns_html(suggestions)
    evidence_graph = _render_evidence_graph_inline(nodes, edges)
    evidence_tables = _build_evidence_html(nodes, edges)
    investigation = _build_investigation_html(case_pack)
    audit_trail = _build_audit_trail_html(traces)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    display_name = scenario_name.replace("_", " ").title()

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AMEX Fraud Simulation Report — {display_name}</title>
  <style>
    /* Reset */
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    body {{
      font-family: 'Helvetica Neue', Arial, sans-serif;
      background: {AMEX_BG};
      color: #333;
      line-height: 1.6;
    }}

    .page {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 20px;
    }}

    {DASHBOARD_CSS}

    /* Print-friendly tweaks */
    @media print {{
      body {{ background: #fff; }}
      .page {{ max-width: 100%; padding: 0; }}
      .section-header {{ break-after: avoid; }}
      .chat-container {{ max-height: none; overflow: visible; }}
      details {{ display: block; }}
      details > summary {{ display: none; }}
      details > div {{ display: block !important; }}
    }}

    /* Additional report styles */
    .report-header {{
      background: linear-gradient(135deg, {AMEX_NAVY} 0%, {AMEX_BLUE} 100%);
      padding: 24px 32px;
      border-radius: 8px;
      margin-bottom: 24px;
      color: {AMEX_WHITE};
    }}
    .report-header h1 {{ font-size: 1.6em; margin-bottom: 4px; }}
    .report-header .meta {{ color: {AMEX_LIGHT_BLUE}; font-size: 0.9em; }}

    .two-col {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }}
    @media (max-width: 768px) {{
      .two-col {{ grid-template-columns: 1fr; }}
    }}

    .collapsible-header {{
      cursor: pointer;
      user-select: none;
    }}
  </style>
</head>
<body>
  <div class="page">

    <!-- Header -->
    <div class="report-header">
      <h1>AMEX Fraud Simulation Report</h1>
      <div class="meta">
        Scenario: <strong>{display_name}</strong> &middot;
        Case: <strong>{case_id or "N/A"}</strong> &middot;
        Generated: {generated_at}
      </div>
    </div>

    <!-- Section 1: Case Overview -->
    <div class="section-header">Case Overview</div>
    {case_overview}

    <!-- Section 2 & 3: Conversation & Copilot -->
    <div class="section-header">Conversation &amp; Copilot Analysis</div>
    <div class="two-col" style="margin-bottom:12px;">
      <div class="card">{hypothesis_chart}</div>
      <div class="card">{eligibility_chart}</div>
    </div>
    {copilot_final}
    <div class="two-col">
      <div>
        <h4 style="color:{AMEX_NAVY}; margin-bottom:8px;">Conversation Transcript</h4>
        {transcript}
      </div>
      <div>
        <h4 style="color:{AMEX_NAVY}; margin-bottom:8px;">Per-Turn Copilot Details</h4>
        {copilot_turns}
      </div>
    </div>

    <!-- Section 4: Evidence Graph -->
    <div class="section-header">Evidence Graph</div>
    {evidence_graph}
    <details style="margin-bottom:16px;">
      <summary style="cursor:pointer; font-weight:600; color:{AMEX_BLUE};
                       padding:10px; background:{AMEX_WHITE}; border:1px solid #E0E4EA;
                       border-radius:6px;">
        Evidence Tables (Detail View)</summary>
      {evidence_tables}
    </details>

    <!-- Section 5: Investigation Results -->
    <div class="section-header">Investigation Results</div>
    {investigation}

    <!-- Section 6: Audit Trail -->
    <details style="margin-bottom:16px;">
      <summary style="cursor:pointer; font-weight:600; color:{AMEX_BLUE};
                       padding:10px; background:{AMEX_WHITE}; border:1px solid #E0E4EA;
                       border-radius:6px;">
        Audit Trail</summary>
      {audit_trail}
    </details>

    <!-- Footer -->
    <div style="text-align:center; padding:20px 0; color:#999; font-size:0.8em;">
      Generated by AMEX Fraud Servicing Simulation System &middot; {generated_at}
    </div>

  </div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments and generate the HTML report."""
    parser = argparse.ArgumentParser(
        description="Export simulation results as a self-contained HTML report."
    )
    parser.add_argument(
        "--scenario",
        "-s",
        help="Name of the scenario directory under data/simulation/",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path (default: reports/{scenario}.html)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available scenarios and exit",
    )
    args = parser.parse_args()

    if args.list:
        scenarios = discover_scenarios(BASE_DIR)
        if not scenarios:
            print("No scenarios found in data/simulation/")
        else:
            print("Available scenarios:")
            for s in scenarios:
                print(f"  - {s}")
        return

    if not args.scenario:
        parser.error("--scenario is required (use --list to see available scenarios)")

    # Validate scenario exists
    db_dir = os.path.join(BASE_DIR, args.scenario)
    if not os.path.isdir(db_dir):
        print(f"Error: scenario directory not found: {db_dir}", file=sys.stderr)
        sys.exit(1)

    # Determine output path
    if args.output:
        out_path = args.output
    else:
        os.makedirs("reports", exist_ok=True)
        out_path = os.path.join("reports", f"{args.scenario}.html")

    print(f"Loading data from {db_dir} ...")
    html = _build_full_report(args.scenario)

    # Ensure parent directory exists
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"Report saved to: {out_path} ({size_kb:.0f} KB)")
    print("Open in any browser — no server needed.")


if __name__ == "__main__":
    main()
