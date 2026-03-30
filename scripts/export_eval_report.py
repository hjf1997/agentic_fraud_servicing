"""Export evaluation results as a self-contained HTML report.

Reads from JSON files under data/evaluations/{scenario}/ and produces a single
.html file that can be opened in any browser — no server needed.

Usage:
    python scripts/export_eval_report.py --scenario scam_techvault
    python scripts/export_eval_report.py --scenario scam_techvault -o ~/Desktop/eval.html
    python scripts/export_eval_report.py --list
"""

import argparse
import base64
import io
import os
import sys
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

# Ensure the project root is importable when running as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agentic_fraud_servicing.evaluation.eval_data_loader import (  # noqa: E402
    discover_eval_scenarios,
    load_evaluation_report,
    load_evaluation_run,
)
from agentic_fraud_servicing.ui.eval_dashboard import (  # noqa: E402
    AMEX_BG,
    AMEX_BLUE,
    AMEX_LIGHT_BLUE,
    AMEX_NAVY,
    AMEX_WHITE,
    EVAL_CSS,
    _build_adherence_chart,
    _build_adherence_detail_html,
    _build_allegation_html,
    _build_decision_html,
    _build_eval_transcript_html,
    _build_evidence_table_html,
    _build_flagged_turns_html,
    _build_hypothesis_chart,
    _build_latency_chart,
    _build_latency_stats_html,
    _build_note_alignment_html,
    _build_prediction_html,
    _build_radar_chart,
    _build_summary_html,
)

BASE_DIR = "data/evaluations"


# ---------------------------------------------------------------------------
# Chart → base64 PNG helpers
# ---------------------------------------------------------------------------


def _fig_to_base64(fig: plt.Figure | None, fallback: str = "No data available.") -> str:
    """Convert a matplotlib Figure to a base64-encoded <img> tag.

    Returns a placeholder paragraph if fig is None.
    """
    if fig is None:
        return f'<p style="color:#666;">{fallback}</p>'

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")

    return f'<img src="data:image/png;base64,{b64}" style="width:100%; max-width:800px;" />'


# ---------------------------------------------------------------------------
# Full HTML report assembly
# ---------------------------------------------------------------------------


def _build_full_report(scenario_name: str) -> str:
    """Load all data for a scenario and assemble a self-contained HTML report."""
    scenario_dir = os.path.join(BASE_DIR, scenario_name)

    run = load_evaluation_run(scenario_dir)
    report = load_evaluation_report(scenario_dir)

    # Build HTML sections from eval_dashboard builders
    summary_html = _build_summary_html(report, run)
    latency_stats_html = _build_latency_stats_html(report)
    flagged_turns_html = _build_flagged_turns_html(report)
    prediction_html = _build_prediction_html(report)
    adherence_detail_html = _build_adherence_detail_html(report)
    allegation_html = _build_allegation_html(report)
    evidence_html = _build_evidence_table_html(report)
    decision_html = _build_decision_html(report)
    note_alignment_html = _build_note_alignment_html(report)
    transcript_html = _build_eval_transcript_html(run)

    # Build charts and render as base64 PNGs
    radar_img = _fig_to_base64(
        _build_radar_chart(report), "No evaluation report available for radar chart."
    )
    latency_img = _fig_to_base64(_build_latency_chart(report), "No latency data available.")
    hypothesis_img = _fig_to_base64(
        _build_hypothesis_chart(run, report), "No hypothesis data available."
    )
    adherence_img = _fig_to_base64(_build_adherence_chart(report), "No adherence data available.")

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    display_name = scenario_name.replace("_", " ").title()
    scenario_id = run.scenario_name if run else scenario_name

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AMEX Evaluation Report — {display_name}</title>
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

    {EVAL_CSS}

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

    /* Report-specific styles */
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

    .section-title {{
      font-size: 1.2em;
      font-weight: bold;
      color: {AMEX_NAVY};
      margin: 24px 0 12px 0;
      padding-bottom: 6px;
      border-bottom: 2px solid {AMEX_BLUE};
    }}
  </style>
</head>
<body>
  <div class="page">

    <!-- Header -->
    <div class="report-header">
      <h1>AMEX Copilot Evaluation Report</h1>
      <div class="meta">
        Scenario: <strong>{display_name}</strong> &middot;
        ID: <strong>{scenario_id}</strong> &middot;
        Generated: {generated_at}
      </div>
    </div>

    <!-- Section 1: Evaluation Summary -->
    <div class="section-title">1. Evaluation Summary</div>
    <div class="two-col" style="margin-bottom:12px;">
      <div>{summary_html}</div>
      <div class="card">{radar_img}</div>
    </div>

    <!-- Section 2: Latency Analysis -->
    <div class="section-title">2. Latency Analysis</div>
    <div class="two-col" style="margin-bottom:12px;">
      <div class="card">{latency_img}</div>
      <div>
        {latency_stats_html}
        {flagged_turns_html}
      </div>
    </div>

    <!-- Section 3: Prediction & Hypothesis Evolution -->
    <div class="section-title">3. Prediction &amp; Hypothesis Evolution</div>
    <div class="two-col" style="margin-bottom:12px;">
      <div class="card">{hypothesis_img}</div>
      <div>{prediction_html}</div>
    </div>

    <!-- Section 4: Question Adherence -->
    <div class="section-title">4. Question Adherence</div>
    <div class="two-col" style="margin-bottom:12px;">
      <div class="card">{adherence_img}</div>
      <div>{adherence_detail_html}</div>
    </div>

    <!-- Section 5: Allegation Extraction Quality -->
    <div class="section-title">5. Allegation Extraction Quality</div>
    {allegation_html}

    <!-- Section 6: Evidence Utilization -->
    <div class="section-title">6. Evidence Utilization</div>
    {evidence_html}

    <!-- Section 7: Decision Explanation & Risk Flags -->
    <div class="section-title">7. Decision Explanation &amp; Risk Flags</div>
    {decision_html}

    <!-- Section 8: CCP Note Alignment -->
    <div class="section-title">8. CCP Note Alignment</div>
    {note_alignment_html}

    <!-- Section 9: Transcript Replay -->
    <details style="margin-bottom:16px;">
      <summary class="section-title" style="cursor:pointer; border-bottom:none;">
        9. Transcript Replay (click to expand)
      </summary>
      {transcript_html}
    </details>

    <!-- Footer -->
    <div style="text-align:center; padding:20px 0; color:#999; font-size:0.8em;">
      Generated by AMEX Fraud Servicing Evaluation System &middot; {generated_at}
    </div>

  </div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments and generate the HTML evaluation report."""
    parser = argparse.ArgumentParser(
        description="Export evaluation results as a self-contained HTML report."
    )
    parser.add_argument(
        "--scenario",
        "-s",
        help="Name of the scenario directory under data/evaluations/",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path (default: data/evaluations/{scenario}/evaluation_report.html)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available evaluation scenarios and exit",
    )
    args = parser.parse_args()

    if args.list:
        scenarios = discover_eval_scenarios(BASE_DIR)
        if not scenarios:
            print("No evaluation scenarios found in data/evaluations/")
        else:
            print("Available evaluation scenarios:")
            for s in scenarios:
                print(f"  - {s}")
        return

    if not args.scenario:
        parser.error("--scenario is required (use --list to see available scenarios)")

    # Validate scenario directory exists
    scenario_dir = os.path.join(BASE_DIR, args.scenario)
    if not os.path.isdir(scenario_dir):
        print(f"Error: scenario directory not found: {scenario_dir}", file=sys.stderr)
        sys.exit(1)

    # Determine output path
    if args.output:
        out_path = args.output
    else:
        out_path = os.path.join(scenario_dir, "evaluation_report.html")

    print(f"Loading evaluation data from {scenario_dir} ...")
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
