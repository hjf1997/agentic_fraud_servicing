"""AMEX-branded Gradio dashboard for evaluation results.

Read-only dashboard that displays copilot evaluation metrics across 8 quality
dimensions. No LLM credentials required — all data is pre-computed by the
evaluation runner and report aggregator.

Entry point: ``python -m agentic_fraud_servicing.ui.eval_dashboard``
"""

from __future__ import annotations

import math
import os

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt

from agentic_fraud_servicing.evaluation.eval_data_loader import (
    discover_eval_scenarios,
    extract_dimension_scores,
    load_evaluation_report,
    load_evaluation_run,
)
from agentic_fraud_servicing.evaluation.models import EvaluationReport, EvaluationRun

matplotlib.use("Agg")

# -- AMEX color palette -----------------------------------------------------------

AMEX_BLUE = "#006FCF"
AMEX_NAVY = "#00175A"
AMEX_LIGHT_BLUE = "#B3E0FF"
AMEX_BG = "#F7F8FA"
AMEX_WHITE = "#FFFFFF"

BASE_DIR = "data/evaluations"

# -- CSS --------------------------------------------------------------------------

EVAL_CSS = """
.gradio-container { font-family: 'Helvetica Neue', Arial, sans-serif; }

.section-header {
    background: linear-gradient(135deg, %(navy)s 0%%, %(blue)s 100%%);
    color: %(white)s; padding: 14px 24px; border-radius: 8px 8px 0 0;
    font-size: 1.25em; font-weight: 700; margin-top: 16px;
    letter-spacing: 0.5px;
}

.card {
    background: %(white)s; border: 1px solid #E0E4EA;
    border-radius: 8px; padding: 20px; margin-bottom: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}

.cat-badge { display: inline-block; padding: 3px 10px; border-radius: 12px;
             font-size: 0.85em; font-weight: 600; }
.cat-third-party  { background: #CCE5FF; color: #004085; }
.cat-first-party  { background: #F8D7DA; color: #721C24; }
.cat-scam         { background: #FFF3CD; color: #856404; }
.cat-dispute      { background: #D4EDDA; color: #155724; }
.cat-undetermined { background: #E2E3E5; color: #383D41; }

.metric-box {
    display: inline-block; text-align: center; padding: 10px 16px;
    border-radius: 8px; background: %(bg)s; min-width: 100px;
}
.metric-value { font-size: 1.8em; font-weight: 700; }
.metric-label { font-size: 0.8em; color: #666; margin-top: 2px; }
""" % {
    "navy": AMEX_NAVY,
    "blue": AMEX_BLUE,
    "bg": AMEX_BG,
    "white": AMEX_WHITE,
}

# -- Dimension display names (order matches radar chart) --------------------------

DIMENSION_LABELS = [
    ("latency", "Latency"),
    ("prediction", "Prediction"),
    ("question_adherence", "Q. Lifecycle"),
    ("allegation_quality", "Allegation"),
    ("evidence_utilization", "Evidence"),
    ("convergence", "Convergence"),
    ("risk_flag_timeliness", "Risk Flags"),
    ("decision_explanation", "Decision"),
]


# -- Helper: category badge -------------------------------------------------------


def _category_badge(category: str) -> str:
    """Render an investigation category as a colored badge."""
    css_map = {
        "THIRD_PARTY_FRAUD": "cat-third-party",
        "FIRST_PARTY_FRAUD": "cat-first-party",
        "SCAM": "cat-scam",
        "DISPUTE": "cat-dispute",
        "UNABLE_TO_DETERMINE": "cat-undetermined",
    }
    css_class = css_map.get(category.upper() if category else "", "")
    return f'<span class="cat-badge {css_class}">{category}</span>'


def _score_color(score: float | None) -> str:
    """Return a hex color for a 0-1 score (green > 0.8, yellow > 0.6, red)."""
    if score is None:
        return "#999"
    if score > 0.8:
        return "#155724"
    if score > 0.6:
        return "#856404"
    return "#721C24"


def _score_bg(score: float | None) -> str:
    """Return a background color for a 0-1 score."""
    if score is None:
        return "#E0E4EA"
    if score > 0.8:
        return "#D4EDDA"
    if score > 0.6:
        return "#FFF3CD"
    return "#F8D7DA"


# -- Section 1: Evaluation Summary ------------------------------------------------


def _build_summary_html(report: EvaluationReport | None, run: EvaluationRun | None) -> str:
    """Build HTML for the Evaluation Summary card.

    Shows overall score, ground truth vs prediction, and key metrics row.
    """
    if not report:
        return '<div class="card"><p>No evaluation report available.</p></div>'

    # Overall score
    overall = report.overall_score
    score_color = _score_color(overall)
    score_bg = _score_bg(overall)

    # Prediction vs ground truth
    predicted = ""
    actual = ""
    match_icon = ""
    if report.prediction:
        predicted = report.prediction.predicted_category
        actual = report.prediction.ground_truth_category
        match_icon = (
            '<span style="color:#155724; font-size:1.4em;">&#10003;</span>'
            if report.prediction.match
            else '<span style="color:#721C24; font-size:1.4em;">&#10007;</span>'
        )

    # Key metrics
    p95 = f"{report.latency.p95_ms:.0f}ms" if report.latency else "N/A"
    compliance = f"{report.latency.compliance_rate * 100:.0f}%" if report.latency else "N/A"
    adherence = (
        f"{report.question_adherence.overall_adherence_rate * 100:.0f}%"
        if report.question_adherence
        else "N/A"
    )
    allegation_f1 = (
        f"{report.allegation_quality.f1_score:.2f}" if report.allegation_quality else "N/A"
    )
    convergence_turn = "N/A"
    if report.convergence:
        ct = report.convergence.convergence_turn
        convergence_turn = f"Turn {ct}" if ct is not None else "Never"

    total_turns = run.total_turns if run else "N/A"

    return f"""<div class="card">
      <div style="display:flex; align-items:flex-start; gap:24px; flex-wrap:wrap;">
        <div style="text-align:center; padding:16px 24px; border-radius:12px;
                    background:{score_bg}; min-width:140px;">
          <div style="font-size:2.8em; font-weight:800; color:{score_color};">
            {overall:.0%}</div>
          <div style="font-size:0.9em; color:#666;">Overall Quality</div>
        </div>

        <div style="flex:1; min-width:280px;">
          <h4 style="color:{AMEX_NAVY}; margin:0 0 10px 0;">
            Ground Truth vs Prediction</h4>
          <div style="display:flex; align-items:center; gap:12px; flex-wrap:wrap;">
            <div>
              <span style="font-size:0.8em; color:#666;">Predicted:</span><br/>
              {_category_badge(predicted) if predicted else "N/A"}
            </div>
            <div style="font-size:1.4em; color:#666;">&rarr;</div>
            <div>
              <span style="font-size:0.8em; color:#666;">Ground Truth:</span><br/>
              {_category_badge(actual) if actual else "N/A"}
            </div>
            <div>{match_icon}</div>
          </div>
          <div style="margin-top:12px; font-size:0.85em; color:#666;">
            Scenario: <strong>{report.scenario_name}</strong>
            &nbsp;|&nbsp; Turns: <strong>{total_turns}</strong>
            &nbsp;|&nbsp; Generated: {report.generated_at[:19] if report.generated_at else "N/A"}
          </div>
        </div>
      </div>

      <div style="display:flex; gap:12px; flex-wrap:wrap; margin-top:16px;
                  justify-content:space-around;">
        <div class="metric-box">
          <div class="metric-value" style="color:{AMEX_BLUE};">{p95}</div>
          <div class="metric-label">Latency P95</div>
        </div>
        <div class="metric-box">
          <div class="metric-value" style="color:{AMEX_BLUE};">{compliance}</div>
          <div class="metric-label">Compliance</div>
        </div>
        <div class="metric-box">
          <div class="metric-value" style="color:{AMEX_BLUE};">{adherence}</div>
          <div class="metric-label">Adherence</div>
        </div>
        <div class="metric-box">
          <div class="metric-value" style="color:{AMEX_BLUE};">{allegation_f1}</div>
          <div class="metric-label">Allegation F1</div>
        </div>
        <div class="metric-box">
          <div class="metric-value" style="color:{AMEX_BLUE};">{convergence_turn}</div>
          <div class="metric-label">Convergence</div>
        </div>
      </div>
    </div>"""


def _build_radar_chart(report: EvaluationReport | None) -> plt.Figure | None:
    """Build a matplotlib radar/spider chart of the 8 evaluation dimensions.

    Dimensions with None scores are plotted as 0.0 with a dashed outline
    to distinguish them from genuinely-zero scores.
    """
    if not report:
        return None

    scores_dict = extract_dimension_scores(report)
    labels = [label for _, label in DIMENSION_LABELS]
    keys = [key for key, _ in DIMENSION_LABELS]
    values = [scores_dict.get(k) for k in keys]

    # Replace None with 0.0 for plotting; track which are missing
    plot_values = [v if v is not None else 0.0 for v in values]
    missing_mask = [v is None for v in values]

    n = len(labels)
    angles = [i * 2 * math.pi / n for i in range(n)]
    angles.append(angles[0])  # close the polygon
    plot_values_closed = plot_values + [plot_values[0]]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"polar": True})
    fig.patch.set_facecolor(AMEX_WHITE)

    # Draw the filled area
    ax.fill(angles, plot_values_closed, color=AMEX_BLUE, alpha=0.15)
    ax.plot(angles, plot_values_closed, color=AMEX_BLUE, linewidth=2)

    # Mark missing dimensions with dashed outline
    for i, is_missing in enumerate(missing_mask):
        if is_missing:
            next_i = (i + 1) % n
            ax.plot(
                [angles[i], angles[next_i]],
                [plot_values_closed[i], plot_values_closed[next_i]],
                color="#999",
                linewidth=1.5,
                linestyle="--",
            )

    # Points
    for i, (val, is_missing) in enumerate(zip(plot_values, missing_mask)):
        color = "#999" if is_missing else AMEX_BLUE
        marker = "o" if not is_missing else "x"
        ax.plot(angles[i], val, marker, color=color, markersize=7)

    # Labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9, color=AMEX_NAVY)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7, color="#999")
    ax.set_title("Evaluation Dimensions", fontsize=12, color=AMEX_NAVY, pad=20)

    plt.tight_layout()
    return fig


# -- Section 2: Latency Analysis --------------------------------------------------


def _build_latency_chart(report: EvaluationReport | None) -> plt.Figure | None:
    """Build a bar chart of per-turn latency with 1500ms compliance threshold."""
    if not report or not report.latency:
        return None

    latency = report.latency
    raw_values = latency.per_turn_latency_ms
    if not raw_values:
        return None

    # Use assessed turn numbers if available, otherwise fall back to 1..N
    raw_turns = getattr(latency, "assessed_turns", None) or list(range(1, len(raw_values) + 1))

    # Only show turns with non-zero latency (turns with 0ms have no real data)
    turns = []
    values = []
    for t, v in zip(raw_turns, raw_values):
        if v > 0:
            turns.append(t)
            values.append(v)

    if not values:
        return None

    threshold = latency.compliance_target_ms
    colors = ["#D32F2F" if v > threshold else "#2E7D32" for v in values]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor(AMEX_WHITE)
    ax.set_facecolor(AMEX_BG)

    ax.bar(turns, values, color=colors, width=0.7, edgecolor="none")
    ax.axhline(
        y=threshold,
        color=AMEX_BLUE,
        linestyle="--",
        linewidth=1.5,
        label=f"Target: {threshold:.0f}ms",
    )

    ax.set_xlabel("Turn (assessed only)", fontsize=10, color=AMEX_NAVY)
    ax.set_ylabel("Latency (ms)", fontsize=10, color=AMEX_NAVY)
    ax.set_title("Per-Turn Copilot Latency", fontsize=12, color=AMEX_NAVY)
    ax.legend(fontsize=9)
    ax.set_xticks(turns)
    if len(turns) > 15:
        ax.tick_params(axis="x", labelsize=7, labelrotation=45)

    plt.tight_layout()
    return fig


def _build_latency_stats_html(
    report: EvaluationReport | None, run: "EvaluationRun | None" = None
) -> str:
    """Build HTML showing latency statistics: p50, p95, p99, max, compliance."""
    if not report or not report.latency:
        return '<div class="card"><p>No latency data available.</p></div>'

    lat = report.latency
    compliance_color = _score_color(lat.compliance_rate)

    return f"""<div class="card">
      <h4 style="color:{AMEX_NAVY}; margin:0 0 12px 0;">Latency Statistics</h4>
      <div style="display:flex; gap:12px; flex-wrap:wrap; justify-content:space-around;">
        <div class="metric-box">
          <div class="metric-value" style="color:{AMEX_BLUE};">{lat.p50_ms:.0f}</div>
          <div class="metric-label">P50 (ms)</div>
        </div>
        <div class="metric-box">
          <div class="metric-value" style="color:{AMEX_BLUE};">{lat.p95_ms:.0f}</div>
          <div class="metric-label">P95 (ms)</div>
        </div>
        <div class="metric-box">
          <div class="metric-value" style="color:{AMEX_BLUE};">{lat.p99_ms:.0f}</div>
          <div class="metric-label">P99 (ms)</div>
        </div>
        <div class="metric-box">
          <div class="metric-value" style="color:{AMEX_BLUE};">{lat.max_ms:.0f}</div>
          <div class="metric-label">Max (ms)</div>
        </div>
        <div class="metric-box">
          <div class="metric-value" style="color:{compliance_color};">
            {lat.compliance_rate * 100:.0f}%</div>
          <div class="metric-label">Compliance (&lt;{lat.compliance_target_ms:.0f}ms)</div>
        </div>
      </div>

      {_build_flagged_turns_html(report, run)}
    </div>"""


def _build_flagged_turns_html(
    report: EvaluationReport | None, run: "EvaluationRun | None" = None
) -> str:
    """Build an HTML table of turns exceeding the latency threshold."""
    if not report or not report.latency or not report.latency.flagged_turns:
        return ""

    lat = report.latency

    # Build turn_number -> latency_ms mapping
    # Prefer assessed_turns from report; fall back to EvaluationRun turn_metrics
    assessed = getattr(lat, "assessed_turns", None) or []
    if assessed and len(assessed) == len(lat.per_turn_latency_ms):
        turn_to_latency = dict(zip(assessed, lat.per_turn_latency_ms))
    elif run and run.turn_metrics:
        turn_to_latency = {
            m.turn_number: m.latency_ms
            for m in run.turn_metrics
            if m.copilot_suggestion is not None
        }
    else:
        turn_to_latency = {}

    rows = ""
    for turn_num in lat.flagged_turns:
        latency_val = turn_to_latency.get(turn_num, 0.0)
        rows += (
            f"<tr>"
            f"<td style='padding:5px 10px;'>Turn {turn_num}</td>"
            f"<td style='padding:5px 10px; color:#D32F2F; font-weight:600;'>"
            f"{latency_val:.0f}ms</td>"
            f"</tr>"
        )

    return f"""
      <h4 style="color:{AMEX_NAVY}; margin:16px 0 8px 0;">
        Flagged Turns ({len(lat.flagged_turns)})</h4>
      <table style="width:100%; border-collapse:collapse; font-size:0.9em;">
        <thead>
          <tr style="background:{AMEX_BLUE}; color:#fff;">
            <th style="padding:5px 10px; text-align:left;">Turn</th>
            <th style="padding:5px 10px; text-align:left;">Latency</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>"""


# -- Section 3: Prediction & Hypothesis Evolution --------------------------------


def _build_hypothesis_chart(
    run: EvaluationRun | None, report: EvaluationReport | None
) -> plt.Figure | None:
    """Build a line chart showing hypothesis score evolution across turns.

    Five colored lines for each InvestigationCategory. Ground truth category
    shown as a light horizontal band. Convergence turn marked with a vertical
    dashed line.
    """
    if not run or not run.turn_metrics:
        return None

    category_colors = {
        "THIRD_PARTY_FRAUD": AMEX_BLUE,
        "FIRST_PARTY_FRAUD": "#D32F2F",
        "SCAM": "#F57C00",
        "DISPUTE": "#2E7D32",
        "UNABLE_TO_DETERMINE": "#97999B",
    }
    categories = list(category_colors.keys())

    # Separate assessed turns (with copilot_suggestion) from non-assessed
    assessed_turns = []
    assessed_series: dict[str, list[float]] = {cat: [] for cat in categories}
    for m in run.turn_metrics:
        if m.copilot_suggestion is not None:
            assessed_turns.append(m.turn_number)
            for cat in categories:
                assessed_series[cat].append(m.hypothesis_scores.get(cat, 0.0))

    if not assessed_turns:
        return None

    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor(AMEX_WHITE)
    ax.set_facecolor(AMEX_BG)

    # Ground truth band
    gt_category = run.ground_truth.get("investigation_category", "")
    if gt_category in category_colors:
        ax.axhspan(0.9, 1.0, color=category_colors[gt_category], alpha=0.10)
        ax.text(
            assessed_turns[-1] + 0.3,
            0.95,
            f"GT: {gt_category}",
            fontsize=7,
            color=category_colors[gt_category],
            va="center",
        )

    # Convergence marker
    if report and report.convergence and report.convergence.convergence_turn is not None:
        ct = report.convergence.convergence_turn
        ax.axvline(x=ct, color="#999", linestyle="--", linewidth=1.2, label=f"Converge @ {ct}")

    # Score lines — only assessed turns
    for cat in categories:
        ax.plot(
            assessed_turns,
            assessed_series[cat],
            color=category_colors[cat],
            linewidth=1.8,
            label=cat.replace("_", " ").title(),
            marker="o",
            markersize=3,
        )

    ax.set_xlabel("Turn (assessed only)", fontsize=10, color=AMEX_NAVY)
    ax.set_ylabel("Score", fontsize=10, color=AMEX_NAVY)
    ax.set_title("Hypothesis Score Evolution", fontsize=12, color=AMEX_NAVY)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xticks(assessed_turns)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)

    plt.tight_layout()
    return fig


def _build_prediction_html(report: EvaluationReport | None) -> str:
    """Build HTML showing prediction vs ground truth with reasoning."""
    if not report or not report.prediction:
        return '<div class="card"><p>No prediction data available.</p></div>'

    pred = report.prediction
    match_icon = (
        '<span style="color:#155724; font-size:1.4em;">&#10003; Match</span>'
        if pred.match
        else '<span style="color:#721C24; font-size:1.4em;">&#10007; Mismatch</span>'
    )
    delta_color = (
        _score_color(1.0 - abs(pred.confidence_delta)) if pred.confidence_delta else "#666"
    )

    reasoning_html = ""
    if pred.reasoning:
        reasoning_html = f"""
        <div style="margin-top:12px; padding:10px; background:{AMEX_BG};
                    border-radius:6px; font-size:0.85em; color:#333;">
          <strong style="color:{AMEX_NAVY};">Reasoning:</strong><br/>
          {pred.reasoning}
        </div>"""

    return f"""<div class="card">
      <h4 style="color:{AMEX_NAVY}; margin:0 0 12px 0;">Prediction Analysis</h4>
      <div style="display:flex; align-items:center; gap:16px; flex-wrap:wrap;">
        <div>
          <span style="font-size:0.8em; color:#666;">Predicted:</span><br/>
          {_category_badge(pred.predicted_category)}
        </div>
        <div style="font-size:1.2em; color:#666;">&rarr;</div>
        <div>
          <span style="font-size:0.8em; color:#666;">Ground Truth:</span><br/>
          {_category_badge(pred.ground_truth_category)}
        </div>
        <div>{match_icon}</div>
      </div>
      <div style="margin-top:10px;">
        <span style="font-size:0.8em; color:#666;">Confidence Delta:</span>
        <span style="font-weight:600; color:{delta_color};">
          {pred.confidence_delta:.3f}</span>
      </div>
      {reasoning_html}
    </div>"""


# -- Section 4: Question Adherence -----------------------------------------------


def _build_adherence_chart(report: EvaluationReport | None) -> plt.Figure | None:
    """Build a donut chart showing probing question lifecycle distribution."""
    if not report or not report.question_adherence:
        return None

    qa = report.question_adherence
    if qa.total_questions == 0:
        return None

    # Build slices for non-zero statuses
    labels = []
    sizes = []
    colors = []
    status_map = [
        ("Answered", qa.answered, "#2E7D32"),
        ("Invalidated", qa.invalidated, "#D32F2F"),
        ("Skipped", qa.skipped, "#53565A"),
        ("Pending", qa.pending, "#F5A623"),
    ]
    for label, count, color in status_map:
        if count > 0:
            labels.append(f"{label} ({count})")
            sizes.append(count)
            colors.append(color)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    fig.patch.set_facecolor(AMEX_WHITE)

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.0f%%",
        startangle=90,
        pctdistance=0.75,
        textprops={"fontsize": 10},
    )
    for t in autotexts:
        t.set_fontweight("bold")
        t.set_color("#ffffff")

    # Donut hole
    centre = plt.Circle((0, 0), 0.50, fc=AMEX_WHITE)
    ax.add_artist(centre)
    ax.text(
        0,
        0,
        f"{qa.total_questions}",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color=AMEX_NAVY,
    )

    ax.set_title("Probing Question Lifecycle", fontsize=12, color=AMEX_NAVY, pad=12)
    plt.tight_layout()
    return fig


def _build_adherence_detail_html(report: EvaluationReport | None) -> str:
    """Build HTML showing probing question lifecycle list with status badges."""
    if not report or not report.question_adherence:
        return '<div class="card"><p>No probing question data available.</p></div>'

    qa = report.question_adherence

    # Summary stats
    suf_badge = (
        '<span style="background:#D4EDDA; color:#2E7D32; padding:2px 8px; '
        'border-radius:10px; font-size:0.85em; font-weight:600;">Yes</span>'
        if qa.information_sufficient
        else '<span style="background:#FFF3CD; color:#856404; padding:2px 8px; '
        'border-radius:10px; font-size:0.85em; font-weight:600;">No</span>'
    )

    stats_html = f"""
    <div style="display:flex; gap:12px; flex-wrap:wrap; margin-bottom:14px;">
      <div class="metric-box">
        <div class="metric-value" style="color:{AMEX_BLUE};">{qa.total_questions}</div>
        <div class="metric-label">Total Questions</div>
      </div>
      <div class="metric-box">
        <div class="metric-value" style="color:#2E7D32;">{qa.answered}</div>
        <div class="metric-label">Answered</div>
      </div>
      <div class="metric-box">
        <div class="metric-value" style="color:#D32F2F;">{qa.invalidated}</div>
        <div class="metric-label">Invalidated</div>
      </div>
      <div class="metric-box">
        <div class="metric-value" style="color:#53565A;">{qa.skipped}</div>
        <div class="metric-label">Skipped</div>
      </div>
      <div class="metric-box">
        <div class="metric-value" style="color:#F5A623;">{qa.pending}</div>
        <div class="metric-label">Pending</div>
      </div>
    </div>
    <div style="margin-bottom:14px; font-size:0.9em;">
      <strong>Information Sufficient:</strong> {suf_badge}
      &nbsp;&nbsp;
      <strong>Answer Rate:</strong> {qa.overall_adherence_rate:.0%}
    </div>"""

    # Status badge colors
    _STATUS_COLORS = {
        "answered": ("#2E7D32", "#D4EDDA"),
        "invalidated": ("#D32F2F", "#F8D7DA"),
        "skipped": ("#53565A", "#E2E3E5"),
        "pending": ("#856404", "#FFF3CD"),
    }

    # Per-question list
    questions_html = ""
    for pq in qa.probing_questions:
        status = pq.get("status", "pending")
        text = pq.get("text", "")
        target = pq.get("target_category", "")
        reason = pq.get("reason", "")
        sc, sbg = _STATUS_COLORS.get(status, ("#53565A", "#E2E3E5"))

        target_html = (
            f'<span style="color:#666; font-size:0.85em; margin-left:4px;">[{target}]</span>'
            if target
            else ""
        )
        reason_html = (
            f'<div style="margin-top:2px; font-size:0.8em; color:#666;">{reason}</div>'
            if reason
            else ""
        )

        questions_html += f"""
        <div style="display:flex; align-items:flex-start; gap:8px; margin-bottom:6px;
                    padding:6px 10px; border:1px solid #E0E4EA; border-radius:6px;">
          <span style="background:{sbg}; color:{sc}; padding:1px 8px; border-radius:10px;
                       font-size:0.8em; font-weight:700; white-space:nowrap; flex-shrink:0;
                       margin-top:2px;">{status}</span>
          <div style="font-size:0.9em;">
            {text}{target_html}
            {reason_html}
          </div>
        </div>"""

    if not questions_html:
        questions_html = (
            '<div style="font-size:0.9em; color:#666;">No probing questions generated.</div>'
        )

    return f"""<div class="card">
      <h4 style="color:{AMEX_NAVY}; margin:0 0 12px 0;">Probing Question Lifecycle</h4>
      {stats_html}
      <div style="max-height:400px; overflow-y:auto;">{questions_html}</div>
    </div>"""


# -- Section 5: Allegation Extraction Quality ------------------------------------


def _build_allegation_html(report: EvaluationReport | None) -> str:
    """Build HTML showing allegation extraction precision/recall/F1 and item lists."""
    if not report or not report.allegation_quality:
        return '<div class="card"><p>No allegation quality data available.</p></div>'

    aq = report.allegation_quality

    def _metric_html(label: str, value: float) -> str:
        """Render a single P/R/F1 metric box."""
        color = _score_color(value)
        bg = _score_bg(value)
        return f"""<div class="metric-box" style="background:{bg};">
          <div class="metric-value" style="color:{color};">{value:.2f}</div>
          <div class="metric-label">{label}</div>
        </div>"""

    metrics_row = f"""
    <div style="display:flex; gap:12px; flex-wrap:wrap; margin-bottom:16px;
                justify-content:space-around;">
      {_metric_html("Precision", aq.precision)}
      {_metric_html("Recall", aq.recall)}
      {_metric_html("F1 Score", aq.f1_score)}
    </div>"""

    # Allegation item lists
    def _item_list(items: list[str], color: str, bg: str, empty_msg: str) -> str:
        if not items:
            return f'<span style="color:#999; font-size:0.85em;">{empty_msg}</span>'
        html = ""
        for item in items:
            html += (
                f'<div style="padding:4px 8px; margin:3px 0; border-radius:4px; '
                f'background:{bg}; color:{color}; font-size:0.85em;">{item}</div>'
            )
        return html

    matched_html = _item_list(aq.matched, "#155724", "#D4EDDA", "None matched")
    missed_html = _item_list(aq.missed, "#721C24", "#F8D7DA", "None missed")
    fp_html = _item_list(aq.false_positives, "#856404", "#FFF3CD", "No false positives")

    return f"""<div class="card">
      <h4 style="color:{AMEX_NAVY}; margin:0 0 12px 0;">Allegation Extraction Quality</h4>
      {metrics_row}

      <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px;">
        <div>
          <h5 style="color:#155724; margin:0 0 6px 0;">
            Matched ({len(aq.matched)})</h5>
          {matched_html}
        </div>
        <div>
          <h5 style="color:#721C24; margin:0 0 6px 0;">
            Missed ({len(aq.missed)})</h5>
          {missed_html}
        </div>
        <div>
          <h5 style="color:#856404; margin:0 0 6px 0;">
            False Positives ({len(aq.false_positives)})</h5>
          {fp_html}
        </div>
      </div>
    </div>"""


# -- Section 6: Evidence Utilization ----------------------------------------------


def _build_evidence_table_html(report: EvaluationReport | None) -> str:
    """Build an HTML table showing evidence utilization with retrieved/referenced indicators."""
    if not report or not report.evidence_utilization:
        return '<div class="card"><p>No evidence utilization data available.</p></div>'

    eu = report.evidence_utilization

    def _coverage_category(ratio: float) -> str:
        """Map a 0-1 coverage ratio to a categorical label."""
        if ratio >= 0.7:
            return "high"
        if ratio >= 0.4:
            return "medium"
        return "low"

    ret_cat = _coverage_category(eu.retrieval_coverage)
    reas_cat = _coverage_category(eu.reasoning_coverage)
    ret_badge = _rating_badge(ret_cat)
    reas_badge = _rating_badge(reas_cat)
    ret_detail = f"{eu.retrieved_nodes}/{eu.total_evidence_nodes} nodes"
    reas_detail = f"{eu.referenced_in_reasoning}/{eu.total_evidence_nodes} nodes"

    # Coverage summary
    summary = f"""
    <div style="display:flex; gap:12px; flex-wrap:wrap; margin-bottom:14px;
                justify-content:space-around;">
      <div class="metric-box">
        <div class="metric-value">{ret_badge}</div>
        <div class="metric-label">Retrieval Coverage</div>
        <div style="font-size:0.8em; color:#666;">{ret_detail}</div>
      </div>
      <div class="metric-box">
        <div class="metric-value">{reas_badge}</div>
        <div class="metric-label">Reasoning Coverage</div>
        <div style="font-size:0.8em; color:#666;">{reas_detail}</div>
      </div>
      <div class="metric-box">
        <div class="metric-value" style="color:{AMEX_BLUE};">{eu.total_evidence_nodes}</div>
        <div class="metric-label">Total Nodes</div>
      </div>
    </div>"""

    # Missed evidence table
    if not eu.missed_evidence:
        missed_html = (
            '<p style="color:#155724; font-size:0.9em; margin-top:10px;">'
            "All evidence nodes were utilized.</p>"
        )
    else:
        rows = ""
        for item in eu.missed_evidence:
            node_id = item.get("node_id", "unknown")
            node_type = item.get("node_type", "unknown")
            source_type = item.get("source_type", "unknown")
            rows += (
                f'<tr style="background:#FFF3CD;">'
                f'<td style="padding:5px 10px;">{node_id}</td>'
                f'<td style="padding:5px 10px;">{node_type}</td>'
                f'<td style="padding:5px 10px;">{source_type}</td>'
                f'<td style="padding:5px 10px; color:#D32F2F;">&#10007;</td>'
                f'<td style="padding:5px 10px; color:#D32F2F;">&#10007;</td>'
                f"</tr>"
            )
        missed_html = f"""
        <h4 style="color:{AMEX_NAVY}; margin:16px 0 8px 0;">
          Missed Evidence ({len(eu.missed_evidence)})</h4>
        <table style="width:100%; border-collapse:collapse; font-size:0.9em;">
          <thead>
            <tr style="background:{AMEX_BLUE}; color:#fff;">
              <th style="padding:5px 10px; text-align:left;">Node ID</th>
              <th style="padding:5px 10px; text-align:left;">Type</th>
              <th style="padding:5px 10px; text-align:left;">Source</th>
              <th style="padding:5px 10px; text-align:left;">Retrieved</th>
              <th style="padding:5px 10px; text-align:left;">Referenced</th>
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>"""

    return f"""<div class="card">
      <h4 style="color:{AMEX_NAVY}; margin:0 0 12px 0;">Evidence Utilization</h4>
      {summary}
      {missed_html}
    </div>"""


# -- Section 7: Decision Explanation + Risk Flag Timeliness -----------------------


def _build_decision_html(report: EvaluationReport | None) -> str:
    """Build HTML showing decision reasoning chain, evidence, suggestions, and risk flags."""
    parts: list[str] = []

    # Decision explanation
    if report and report.decision_explanation:
        de = report.decision_explanation

        # Reasoning chain
        if de.reasoning_chain:
            chain_html = de.reasoning_chain.replace("\n", "<br/>")
            parts.append(
                f'<div style="margin-bottom:14px;">'
                f'<h4 style="color:{AMEX_NAVY}; margin:0 0 8px 0;">Reasoning Chain</h4>'
                f'<div style="padding:10px; background:{AMEX_BG}; border-radius:6px; '
                f'font-size:0.9em; line-height:1.6; color:#333;">{chain_html}</div>'
                f"</div>"
            )

        # Influential evidence (top 3)
        if de.influential_evidence:
            items = ""
            for ev in de.influential_evidence[:3]:
                evidence = ev.get("evidence", "?")
                influence = ev.get("influence", "")
                description = ev.get("description", "")
                influence_str = (
                    f' <span style="color:{AMEX_BLUE};">[{influence}]</span>' if influence else ""
                )
                desc_str = (
                    f' <span style="color:#666;">{description}</span>' if description else ""
                )
                items += (
                    f'<li style="margin:4px 0;"><strong>{evidence}</strong>'
                    f"{influence_str}{desc_str}</li>"
                )
            parts.append(
                f'<div style="margin-bottom:14px;">'
                f'<h4 style="color:{AMEX_NAVY}; margin:0 0 8px 0;">Influential Evidence</h4>'
                f'<ul style="margin:0; padding-left:20px;">{items}</ul>'
                f"</div>"
            )

        # Improvement suggestions
        if de.improvement_suggestions:
            suggestions = ""
            for i, s in enumerate(de.improvement_suggestions, 1):
                suggestions += f'<li style="margin:4px 0;">{s}</li>'
            parts.append(
                f'<div style="margin-bottom:14px;">'
                f'<h4 style="color:{AMEX_NAVY}; margin:0 0 8px 0;">Improvement Suggestions</h4>'
                f'<ol style="margin:0; padding-left:20px; font-size:0.9em;">{suggestions}</ol>'
                f"</div>"
            )

        # Overall quality notes
        if de.overall_quality_notes:
            notes_html = de.overall_quality_notes.replace("\n", "<br/>")
            parts.append(
                f'<div style="margin-bottom:14px;">'
                f'<h4 style="color:{AMEX_NAVY}; margin:0 0 8px 0;">Quality Notes</h4>'
                f'<div style="padding:10px; background:{AMEX_BG}; border-radius:6px; '
                f'font-size:0.85em; color:#555;">{notes_html}</div>'
                f"</div>"
            )
    else:
        parts.append('<p style="color:#999;">No decision explanation data available.</p>')

    # Risk flag timeliness
    if report and report.risk_flag_timeliness:
        rf = report.risk_flag_timeliness
        parts.append(
            f'<h4 style="color:{AMEX_NAVY}; margin:16px 0 8px 0;">Risk Flag Timeliness</h4>'
        )

        # Stats summary
        parts.append(
            f'<div style="display:flex; gap:12px; flex-wrap:wrap; margin-bottom:10px;">'
            f'<div class="metric-box">'
            f'<div class="metric-value" style="color:{AMEX_BLUE};">'
            f"{rf.flags_raised_count}</div>"
            f'<div class="metric-label">Raised</div></div>'
            f'<div class="metric-box">'
            f'<div class="metric-value" style="color:{AMEX_BLUE};">'
            f"{rf.flags_expected_count}</div>"
            f'<div class="metric-label">Expected</div></div>'
            f'<div class="metric-box">'
            f'<div class="metric-value" style="color:{AMEX_BLUE};">'
            f"{rf.average_delay_turns:.1f}</div>"
            f'<div class="metric-label">Avg Delay (turns)</div></div>'
            f"</div>"
        )

        # Per-flag timing table
        if rf.per_flag_timing:
            rows = ""
            for entry in rf.per_flag_timing:
                flag = entry.get("flag", entry.get("expected_flag", "?"))
                raised = entry.get("raised_turn", "N/A")
                available = entry.get("evidence_available_turn", "N/A")
                delay = entry.get("delay_turns", "N/A")
                delay_color = "#155724" if delay == 0 else "#856404" if delay != "N/A" else "#999"
                rows += (
                    f"<tr>"
                    f'<td style="padding:5px 10px; font-size:0.85em;">{flag}</td>'
                    f'<td style="padding:5px 10px; text-align:center;">{raised}</td>'
                    f'<td style="padding:5px 10px; text-align:center;">{available}</td>'
                    f'<td style="padding:5px 10px; text-align:center; color:{delay_color}; '
                    f'font-weight:600;">{delay}</td>'
                    f"</tr>"
                )
            parts.append(
                f'<table style="width:100%; border-collapse:collapse; font-size:0.9em;">'
                f"<thead>"
                f'<tr style="background:{AMEX_BLUE}; color:#fff;">'
                f'<th style="padding:5px 10px; text-align:left;">Flag</th>'
                f'<th style="padding:5px 10px; text-align:center;">Raised Turn</th>'
                f'<th style="padding:5px 10px; text-align:center;">Evidence Avail.</th>'
                f'<th style="padding:5px 10px; text-align:center;">Delay</th>'
                f"</tr></thead>"
                f"<tbody>{rows}</tbody>"
                f"</table>"
            )
    else:
        parts.append(
            '<p style="color:#999; margin-top:14px;">No risk flag timeliness data available.</p>'
        )

    return f'<div class="card">{chr(10).join(parts)}</div>'


# -- Section 8: CCP Note Alignment ------------------------------------------------


def _rating_badge(label: str) -> str:
    """Return an HTML badge for a categorical rating (low/medium/high)."""
    colors = {
        "high": ("#155724", "#D4EDDA"),
        "medium": ("#856404", "#FFF3CD"),
        "low": ("#721C24", "#F8D7DA"),
    }
    fg, bg = colors.get(label.lower(), ("#333", "#E0E0E0"))
    return (
        f'<span style="display:inline-block; padding:4px 14px; border-radius:12px; '
        f'font-weight:600; font-size:0.95em; background:{bg}; color:{fg};">'
        f"{label.capitalize()}</span>"
    )


def _build_note_alignment_html(report: EvaluationReport | None) -> str:
    """Build HTML showing CCP note alignment ratings and explanation."""
    if not report or not report.note_alignment:
        return '<div class="card"><p>No CCP note alignment data available.</p></div>'

    na = report.note_alignment

    dims = [
        ("Facts Coverage", na.facts_coverage),
        ("Allegation Alignment", na.allegation_alignment),
        ("Category & Action", na.category_action),
        ("Overall", na.overall),
    ]

    metrics_html = ""
    for label, rating in dims:
        metrics_html += (
            f'<div class="metric-box">'
            f'<div class="metric-value">{_rating_badge(rating)}</div>'
            f'<div class="metric-label">{label}</div>'
            f"</div>"
        )

    explanation_html = ""
    if na.explanation:
        explanation_html = (
            f'<div style="margin-top:14px; padding:10px; background:{AMEX_BG}; '
            f'border-radius:6px; font-size:0.85em; line-height:1.6; color:#333;">'
            f'<strong style="color:{AMEX_NAVY};">Explanation:</strong><br/>'
            f"{na.explanation.replace(chr(10), '<br/>')}"
            f"</div>"
        )

    return f"""<div class="card">
      <h4 style="color:{AMEX_NAVY}; margin:0 0 12px 0;">CCP Note Alignment</h4>
      <div style="display:flex; gap:12px; flex-wrap:wrap; justify-content:space-around;">
        {metrics_html}
      </div>
      {explanation_html}
    </div>"""


# -- Section 8: Transcript Replay -------------------------------------------------


def _build_eval_transcript_html(run: EvaluationRun | None) -> str:
    """Build HTML with chat bubbles and per-turn annotations for transcript replay."""
    if not run or not run.turn_metrics:
        return '<div class="card"><p>No transcript data available.</p></div>'

    bubbles = ""
    for m in run.turn_metrics:
        speaker = m.speaker.upper() if m.speaker else "UNKNOWN"
        text = m.text or ""

        # Bubble styling by speaker
        if speaker == "CCP":
            bubble_style = (
                f"background:{AMEX_BLUE}; color:#fff; margin-left:auto; "
                "margin-right:0; text-align:right; border-radius:12px 12px 0 12px;"
            )
            label_style = "text-align:right;"
        elif speaker in ("CARDMEMBER", "CM"):
            bubble_style = (
                "background:#E0E4EA; color:#333; margin-right:auto; "
                "margin-left:0; border-radius:12px 12px 12px 0;"
            )
            label_style = "text-align:left;"
        else:
            bubble_style = (
                f"background:{AMEX_LIGHT_BLUE}; color:{AMEX_NAVY}; "
                "margin:0 auto; text-align:center; border-radius:12px;"
            )
            label_style = "text-align:center;"

        # Top hypothesis for this turn
        scores = m.hypothesis_scores
        top_cat = ""
        top_score = 0.0
        if scores:
            top_cat = max(scores, key=scores.get)
            top_score = scores[top_cat]

        n_allegations = len(m.allegations_extracted)

        # Annotation row below bubble
        is_assessed = m.copilot_suggestion is not None
        annotation = (
            f'<div style="font-size:0.75em; color:#888; {label_style} margin-bottom:10px;">'
            f"Turn {m.turn_number}"
        )
        if is_assessed:
            if top_cat:
                annotation += (
                    f" &bull; {_category_badge(top_cat)}"
                    f' <span style="font-weight:600;">{top_score:.2f}</span>'
                )
            if n_allegations:
                annotation += f" &bull; {n_allegations} allegation(s)"
            annotation += f" &bull; {m.latency_ms:.0f}ms"
        else:
            annotation += ' &bull; <span style="color:#bbb;">not assessed</span>'
        annotation += "</div>"

        bubbles += (
            f'<div style="font-size:0.75em; color:#999; {label_style} '
            f'margin-bottom:2px;"><strong>{speaker}</strong></div>'
            f'<div style="max-width:75%; padding:10px 14px; {bubble_style} '
            f'margin-bottom:4px; font-size:0.9em; line-height:1.4;">{text}</div>'
            f"{annotation}"
        )

    return f"""<div class="card">
      <h4 style="color:{AMEX_NAVY}; margin:0 0 12px 0;">Transcript Replay</h4>
      <div style="max-height:600px; overflow-y:auto; padding:8px;">
        {bubbles}
      </div>
    </div>"""


# -- Main load callback -----------------------------------------------------------


def _load_scenario(scenario_name: str) -> tuple:
    """Load evaluation data and return component updates.

    Returns a tuple of 13 items matching output components.
    """
    if not scenario_name:
        empty = '<div class="card"><p>Select a scenario and click Load.</p></div>'
        return (
            empty,
            None,
            None,
            empty,
            None,
            empty,
            None,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            None,
        )

    scenario_dir = os.path.join(BASE_DIR, scenario_name)
    report = load_evaluation_report(scenario_dir)
    run = load_evaluation_run(scenario_dir)

    return (
        # Section 1
        _build_summary_html(report, run),
        _build_radar_chart(report),
        # Section 2
        _build_latency_chart(report),
        _build_latency_stats_html(report, run),
        # Section 3
        _build_hypothesis_chart(run, report),
        _build_prediction_html(report),
        # Section 4
        _build_adherence_chart(report),
        _build_adherence_detail_html(report),
        # Section 5
        _build_allegation_html(report),
        # Section 6
        _build_evidence_table_html(report),
        # Section 7
        _build_decision_html(report),
        # Section 8
        _build_note_alignment_html(report),
        # Section 9
        _build_eval_transcript_html(run),
        None,  # placeholder for hypothesis chart duplicate avoided
    )


# -- App factory ------------------------------------------------------------------


def create_eval_dashboard_app() -> gr.Blocks:
    """Create the AMEX-branded evaluation results dashboard.

    Returns:
        A configured gr.Blocks instance ready to launch.
    """
    scenarios = discover_eval_scenarios(BASE_DIR)

    with gr.Blocks(title="AMEX Copilot Evaluation Dashboard") as app:
        gr.HTML(f"<style>{EVAL_CSS}</style>")

        # Top bar
        gr.HTML(
            f"""<div style="background: linear-gradient(135deg, {AMEX_NAVY}, {AMEX_BLUE});
                 padding: 16px 24px; border-radius: 8px; margin-bottom: 16px;">
              <h1 style="color:#fff; margin:0; font-size:1.5em;">
                AMEX Copilot Evaluation Dashboard</h1>
              <p style="color:{AMEX_LIGHT_BLUE}; margin:4px 0 0 0; font-size:0.9em;">
                8-dimension quality assessment of copilot performance</p>
            </div>"""
        )

        with gr.Row():
            scenario_dropdown = gr.Dropdown(
                choices=scenarios,
                label="Evaluation Scenario",
                scale=3,
            )
            load_btn = gr.Button("Load Results", variant="primary", scale=1)

        # Section 1: Evaluation Summary
        gr.HTML('<div class="section-header" style="color:white;">Evaluation Summary</div>')
        with gr.Row():
            with gr.Column(scale=3):
                summary_html = gr.HTML()
            with gr.Column(scale=2):
                radar_plot = gr.Plot(label="Dimension Scores")

        # Section 2: Latency Analysis
        gr.HTML('<div class="section-header" style="color:white;">Latency Analysis</div>')
        with gr.Row():
            with gr.Column(scale=3):
                latency_plot = gr.Plot(label="Per-Turn Latency")
            with gr.Column(scale=2):
                latency_stats_html = gr.HTML()

        # Section 3: Prediction & Hypothesis Evolution
        gr.HTML(
            '<div class="section-header" style="color:white;">'
            "Prediction &amp; Hypothesis Evolution</div>"
        )
        with gr.Row():
            with gr.Column(scale=3):
                hypothesis_plot = gr.Plot(label="Hypothesis Scores")
            with gr.Column(scale=2):
                prediction_html = gr.HTML()

        # Section 4: Probing Question Lifecycle
        gr.HTML(
            '<div class="section-header" style="color:white;">Probing Question Lifecycle</div>'
        )
        with gr.Row():
            with gr.Column(scale=2):
                adherence_plot = gr.Plot(label="Lifecycle Distribution")
            with gr.Column(scale=3):
                adherence_detail_html = gr.HTML()

        # Section 5: Allegation Extraction Quality
        gr.HTML(
            '<div class="section-header" style="color:white;">Allegation Extraction Quality</div>'
        )
        allegation_html = gr.HTML()

        # Section 6: Evidence Utilization
        gr.HTML('<div class="section-header" style="color:white;">Evidence Utilization</div>')
        evidence_html = gr.HTML()

        # Section 7: Decision Explanation & Risk Flags
        gr.HTML(
            '<div class="section-header" style="color:white;">'
            "Decision Explanation &amp; Risk Flags</div>"
        )
        decision_html = gr.HTML()

        # Section 8: CCP Note Alignment
        gr.HTML('<div class="section-header" style="color:white;">CCP Note Alignment</div>')
        note_alignment_html = gr.HTML()

        # Section 9: Transcript Replay
        gr.HTML('<div class="section-header" style="color:white;">Transcript Replay</div>')
        transcript_html = gr.HTML()
        transcript_placeholder = gr.Plot(label="", visible=False)

        # Wire load callback
        load_btn.click(
            fn=_load_scenario,
            inputs=[scenario_dropdown],
            outputs=[
                summary_html,
                radar_plot,
                latency_plot,
                latency_stats_html,
                hypothesis_plot,
                prediction_html,
                adherence_plot,
                adherence_detail_html,
                allegation_html,
                evidence_html,
                decision_html,
                note_alignment_html,
                transcript_html,
                transcript_placeholder,
            ],
        )

    return app


def main() -> None:
    """Launch the evaluation dashboard."""
    app = create_eval_dashboard_app()
    app.launch(share=True)


if __name__ == "__main__":
    main()
