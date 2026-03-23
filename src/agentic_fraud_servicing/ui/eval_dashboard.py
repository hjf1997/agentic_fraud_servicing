"""AMEX-branded Gradio dashboard for evaluation results.

Read-only dashboard that displays copilot evaluation metrics across 8 quality
dimensions. No LLM credentials required — all data is pre-computed by the
evaluation runner and report aggregator.

Entry point: ``python -m agentic_fraud_servicing.ui.eval_dashboard``
"""

from __future__ import annotations

import math

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
.cat-third-party { background: #CCE5FF; color: #004085; }
.cat-first-party { background: #F8D7DA; color: #721C24; }
.cat-scam        { background: #FFF3CD; color: #856404; }
.cat-dispute     { background: #D4EDDA; color: #155724; }

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
    ("question_adherence", "Q. Adherence"),
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
    values = latency.per_turn_latency_ms
    if not values:
        return None

    turns = list(range(1, len(values) + 1))
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

    ax.set_xlabel("Turn", fontsize=10, color=AMEX_NAVY)
    ax.set_ylabel("Latency (ms)", fontsize=10, color=AMEX_NAVY)
    ax.set_title("Per-Turn Copilot Latency", fontsize=12, color=AMEX_NAVY)
    ax.legend(fontsize=9)
    ax.set_xticks(turns)

    plt.tight_layout()
    return fig


def _build_latency_stats_html(report: EvaluationReport | None) -> str:
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

      {_build_flagged_turns_html(report)}
    </div>"""


def _build_flagged_turns_html(report: EvaluationReport | None) -> str:
    """Build an HTML table of turns exceeding the latency threshold."""
    if not report or not report.latency or not report.latency.flagged_turns:
        return ""

    lat = report.latency
    rows = ""
    for turn_idx in lat.flagged_turns:
        latency_val = (
            lat.per_turn_latency_ms[turn_idx] if turn_idx < len(lat.per_turn_latency_ms) else 0.0
        )
        rows += (
            f"<tr>"
            f"<td style='padding:5px 10px;'>Turn {turn_idx + 1}</td>"
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

    Four colored lines for each InvestigationCategory. Ground truth category
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
    }
    categories = list(category_colors.keys())

    turns = [m.turn_number for m in run.turn_metrics]
    series: dict[str, list[float]] = {cat: [] for cat in categories}
    for m in run.turn_metrics:
        for cat in categories:
            series[cat].append(m.hypothesis_scores.get(cat, 0.0))

    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor(AMEX_WHITE)
    ax.set_facecolor(AMEX_BG)

    # Ground truth band
    gt_category = run.ground_truth.get("investigation_category", "")
    if gt_category in category_colors:
        ax.axhspan(0.9, 1.0, color=category_colors[gt_category], alpha=0.10)
        ax.text(
            turns[-1] + 0.3,
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

    # Score lines
    for cat in categories:
        ax.plot(
            turns,
            series[cat],
            color=category_colors[cat],
            linewidth=1.8,
            label=cat.replace("_", " ").title(),
            marker="o",
            markersize=3,
        )

    ax.set_xlabel("Turn", fontsize=10, color=AMEX_NAVY)
    ax.set_ylabel("Score", fontsize=10, color=AMEX_NAVY)
    ax.set_title("Hypothesis Score Evolution", fontsize=12, color=AMEX_NAVY)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xticks(turns)
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
    """Build a bar chart of per-turn question adherence scores."""
    if not report or not report.question_adherence:
        return None

    qa = report.question_adherence
    if not qa.per_turn_scores:
        return None

    turns = [s.get("turn", i + 1) for i, s in enumerate(qa.per_turn_scores)]
    scores = [s.get("score", 0.0) for s in qa.per_turn_scores]
    colors = ["#2E7D32" if v >= 0.7 else "#F57C00" if v >= 0.4 else "#D32F2F" for v in scores]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor(AMEX_WHITE)
    ax.set_facecolor(AMEX_BG)

    ax.bar(turns, scores, color=colors, width=0.7, edgecolor="none")
    ax.axhline(y=0.7, color=AMEX_BLUE, linestyle="--", linewidth=1, alpha=0.5)

    ax.set_xlabel("Turn", fontsize=10, color=AMEX_NAVY)
    ax.set_ylabel("Adherence Score", fontsize=10, color=AMEX_NAVY)
    ax.set_title("Per-Turn Question Adherence", fontsize=12, color=AMEX_NAVY)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(turns)

    # Overall rate annotation
    rate_text = f"Overall: {qa.overall_adherence_rate:.0%}"
    ax.annotate(
        rate_text,
        xy=(0.98, 0.95),
        xycoords="axes fraction",
        fontsize=10,
        fontweight="bold",
        color=AMEX_NAVY,
        ha="right",
        va="top",
        bbox={"facecolor": AMEX_WHITE, "edgecolor": AMEX_BLUE, "boxstyle": "round,pad=0.3"},
    )

    plt.tight_layout()
    return fig


def _build_adherence_detail_html(report: EvaluationReport | None) -> str:
    """Build HTML with expandable per-turn adherence details."""
    if not report or not report.question_adherence:
        return '<div class="card"><p>No question adherence data available.</p></div>'

    qa = report.question_adherence

    # Summary stats
    stats_html = f"""
    <div style="display:flex; gap:12px; flex-wrap:wrap; margin-bottom:14px;">
      <div class="metric-box">
        <div class="metric-value" style="color:{_score_color(qa.overall_adherence_rate)};">
          {qa.overall_adherence_rate:.0%}</div>
        <div class="metric-label">Overall Adherence</div>
      </div>
      <div class="metric-box">
        <div class="metric-value" style="color:{AMEX_BLUE};">{qa.turns_with_suggestions}</div>
        <div class="metric-label">Turns w/ Suggestions</div>
      </div>
      <div class="metric-box">
        <div class="metric-value" style="color:{AMEX_BLUE};">{qa.turns_with_adherence}</div>
        <div class="metric-label">Turns w/ Adherence</div>
      </div>
    </div>"""

    # Per-turn collapsible details
    turns_html = ""
    for entry in qa.per_turn_scores:
        turn = entry.get("turn", "?")
        score = entry.get("score", 0.0)
        explanation = entry.get("explanation", "")
        questions = entry.get("questions", [])
        sc = _score_color(score)
        sbg = _score_bg(score)

        q_list = ""
        if questions:
            q_items = "".join(f"<li>{q}</li>" for q in questions)
            q_list = f'<ul style="margin:4px 0; padding-left:20px;">{q_items}</ul>'

        explanation_text = ""
        if explanation:
            explanation_text = (
                f'<div style="margin-top:4px; font-size:0.85em; color:#555;">{explanation}</div>'
            )

        turns_html += f"""
        <details style="margin-bottom:6px; border:1px solid #E0E4EA; border-radius:6px;">
          <summary style="padding:8px 12px; cursor:pointer; background:{AMEX_BG};
                         border-radius:6px; font-size:0.9em;">
            <strong>Turn {turn}</strong>
            <span style="float:right; background:{sbg}; color:{sc}; padding:2px 8px;
                         border-radius:10px; font-size:0.85em; font-weight:600;">
              {score:.2f}</span>
          </summary>
          <div style="padding:8px 12px; font-size:0.85em;">
            {q_list}
            {explanation_text}
          </div>
        </details>"""

    return f"""<div class="card">
      <h4 style="color:{AMEX_NAVY}; margin:0 0 12px 0;">Question Adherence Detail</h4>
      {stats_html}
      <div style="max-height:300px; overflow-y:auto;">{turns_html}</div>
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


# -- Main load callback -----------------------------------------------------------


def _load_scenario(scenario_name: str) -> tuple:
    """Load evaluation data and return component updates.

    Returns a tuple of 10 items matching output components.
    """
    if not scenario_name:
        empty = '<div class="card"><p>Select a scenario and click Load.</p></div>'
        return empty, None, None, empty, None, empty, None, empty, empty, None

    import os

    scenario_dir = os.path.join(BASE_DIR, scenario_name)
    report = load_evaluation_report(scenario_dir)
    run = load_evaluation_run(scenario_dir)

    return (
        # Section 1
        _build_summary_html(report, run),
        _build_radar_chart(report),
        # Section 2
        _build_latency_chart(report),
        _build_latency_stats_html(report),
        # Section 3
        _build_hypothesis_chart(run, report),
        _build_prediction_html(report),
        # Section 4
        _build_adherence_chart(report),
        _build_adherence_detail_html(report),
        # Section 5
        _build_allegation_html(report),
        None,  # placeholder for future sections
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

        # Section 4: Question Adherence
        gr.HTML('<div class="section-header" style="color:white;">Question Adherence</div>')
        with gr.Row():
            with gr.Column(scale=3):
                adherence_plot = gr.Plot(label="Per-Turn Adherence")
            with gr.Column(scale=2):
                adherence_detail_html = gr.HTML()

        # Section 5: Allegation Extraction Quality
        gr.HTML(
            '<div class="section-header" style="color:white;">Allegation Extraction Quality</div>'
        )
        with gr.Row():
            with gr.Column(scale=3):
                allegation_html = gr.HTML()
            with gr.Column(scale=2):
                allegation_placeholder = gr.Plot(label="")

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
                allegation_placeholder,
            ],
        )

    return app


def main() -> None:
    """Launch the evaluation dashboard."""
    app = create_eval_dashboard_app()
    app.launch(share=True)


if __name__ == "__main__":
    main()
