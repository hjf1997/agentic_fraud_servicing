import React, { useState, useEffect, useRef, useCallback } from "react";

/**
 * CCP Desktop Replay — loads simulation data from the API and replays the
 * conversation turn-by-turn with copilot panels updating progressively.
 *
 * Reuses the same 3-panel layout as DemoPage but with dynamic data from
 * SQLite simulation stores served via server.js.
 */

// --- Types ---

type Message = {
  speaker: "CM" | "CCP" | "SYSTEM";
  text: string;
  turn: number;
  run?: number; // which copilot run this turn triggered
};

type ProbingQuestion = {
  text: string;
  status: "pending" | "answered" | "invalidated" | "skipped";
  target: string;
};

type CopilotResult = {
  runLabel: string;
  turnRange: string;
  probingQuestions: ProbingQuestion[];
  informationSufficient: boolean;
  caseEligibility: {
    fraudCase: "eligible" | "blocked" | "incomplete";
    disputeCase: "eligible" | "blocked" | "incomplete";
  };
};

type ScorePoint = { label: string; scores: Record<string, number> };

type RawSuggestion = {
  turn: number;
  phase?: string;
  suggestion: {
    hypothesis_scores?: Record<string, number>;
    probing_questions?: Array<{
      text: string;
      status: string;
      target_category?: string;
    }>;
    suggested_questions?: string[];
    information_sufficient?: boolean;
    case_eligibility?: Array<{
      case_type: string;
      eligibility: string;
    }>;
  };
};

type RawTurn = {
  turn: number;
  speaker: string;
  text: string;
};

type ScenarioData = {
  case: Record<string, unknown> | null;
  turns: RawTurn[];
  suggestions: RawSuggestion[];
  finalState: Record<string, unknown> | null;
};

// --- Score category mapping ---

const SCORE_KEY_MAP: Record<string, string> = {
  THIRD_PARTY_FRAUD: "Fraud",
  FIRST_PARTY_FRAUD: "Bogus",
  SCAM: "Scam",
  DISPUTE: "Billing Dispute",
};

const categoryColors: Record<string, string> = {
  "Fraud": "#006FCF",
  "Bogus": "#F5A623",
  "Scam": "#CF291D",
  "Billing Dispute": "#008000",
};

const categories = Object.keys(categoryColors);

// --- Data transforms ---

function transformData(data: ScenarioData): {
  messages: Message[];
  copilotResults: CopilotResult[];
  scoreHistory: ScorePoint[];
} {
  const { turns, suggestions } = data;

  // Build a set of turn numbers that have copilot suggestions
  const suggestionTurnSet = new Set(suggestions.map((s) => s.turn));

  // Map raw turns to messages, marking which ones trigger a copilot run
  let runIndex = 0;
  const messages: Message[] = turns.map((t) => {
    const speaker = t.speaker === "CARDMEMBER" ? "CM" : t.speaker === "CCP" ? "CCP" : "SYSTEM";
    const msg: Message = { speaker, text: t.text, turn: t.turn };
    if (suggestionTurnSet.has(t.turn)) {
      msg.run = runIndex;
      runIndex++;
    }
    return msg;
  });

  // Transform suggestions into CopilotResult format
  const copilotResults: CopilotResult[] = suggestions.map((s, idx) => {
    const sug = s.suggestion;

    // Map probing questions
    const probingQuestions: ProbingQuestion[] = (sug.probing_questions || []).map((pq) => ({
      text: pq.text,
      status: pq.status as ProbingQuestion["status"],
      target: pq.target_category || "",
    }));

    // If no probing questions but suggested_questions exist, show those as pending
    if (probingQuestions.length === 0 && sug.suggested_questions) {
      for (const q of sug.suggested_questions.slice(0, 3)) {
        probingQuestions.push({ text: q, status: "pending", target: "" });
      }
    }

    // Extract case eligibility
    const ceList = sug.case_eligibility || [];
    let fraudCase: "eligible" | "blocked" | "incomplete" = "incomplete";
    let disputeCase: "eligible" | "blocked" | "incomplete" = "incomplete";
    for (const ce of ceList) {
      const ct = (ce.case_type || "").toLowerCase();
      const elig = (ce.eligibility || "incomplete") as "eligible" | "blocked" | "incomplete";
      if (ct.includes("fraud")) fraudCase = elig;
      if (ct.includes("dispute")) disputeCase = elig;
    }

    return {
      runLabel: `Copilot Run ${idx + 1}`,
      turnRange: `Turn ${s.turn}`,
      probingQuestions,
      informationSufficient: sug.information_sufficient || false,
      caseEligibility: { fraudCase, disputeCase },
    };
  });

  // Build score history from hypothesis scores
  const scoreHistory: ScorePoint[] = [
    { label: "Prior", scores: { "Fraud": 0.25, "Bogus": 0.25, "Scam": 0.25, "Billing Dispute": 0.25 } },
  ];
  for (let i = 0; i < suggestions.length; i++) {
    const hScores = suggestions[i].suggestion.hypothesis_scores || {};
    const mapped: Record<string, number> = {};
    for (const [key, displayName] of Object.entries(SCORE_KEY_MAP)) {
      mapped[displayName] = hScores[key] || 0;
    }
    scoreHistory.push({ label: `R${i + 1}`, scores: mapped });
  }

  return { messages, copilotResults, scoreHistory };
}

// --- Components (matching DemoPage styling) ---

function ScoreCurveChart({
  visibleRuns,
  scoreHistory,
}: {
  visibleRuns: number;
  scoreHistory: ScorePoint[];
}) {
  const pointCount = Math.min(visibleRuns + 1, scoreHistory.length);
  const visibleData = scoreHistory.slice(0, pointCount);

  const W = 800;
  const H = 160;
  const pad = { top: 20, right: 20, bottom: 30, left: 40 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  const xStep = pointCount > 1 ? plotW / (scoreHistory.length - 1) : plotW / 2;
  const toX = (i: number) => pad.left + i * xStep;
  const toY = (v: number) => pad.top + plotH - v * plotH;

  return (
    <div className="score-curve-container">
      <div className="copilot-section-title">Case Typing Score Trend</div>
      <svg viewBox={`0 0 ${W} ${H}`} className="score-curve-svg">
        {[0, 0.25, 0.5, 0.75, 1].map((v) => (
          <g key={v}>
            <line
              x1={pad.left} y1={toY(v)} x2={W - pad.right} y2={toY(v)}
              stroke="#E0E0E0" strokeWidth={1} strokeDasharray={v === 0 ? "0" : "4 4"}
            />
            <text x={pad.left - 6} y={toY(v) + 4} textAnchor="end" fill="#53565A" fontSize={10}>
              {Math.round(v * 100)}%
            </text>
          </g>
        ))}
        {scoreHistory.map((p, i) => (
          <text
            key={i} x={toX(i)} y={H - 6}
            textAnchor="middle" fill={i < pointCount ? "#53565A" : "#C8C9C7"} fontSize={11} fontWeight={600}
          >
            {p.label}
          </text>
        ))}
        {categories.map((cat) => {
          const color = categoryColors[cat];
          const points = visibleData.map((p, i) => ({ x: toX(i), y: toY(p.scores[cat] || 0) }));
          let pathD = "";
          if (points.length > 1) {
            pathD = points.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x} ${p.y}`).join(" ");
          }
          return (
            <g key={cat}>
              {pathD && (
                <path d={pathD} fill="none" stroke={color} strokeWidth={2.5}
                  strokeLinecap="round" strokeLinejoin="round"
                  style={{ transition: "d 0.8s ease" }}
                />
              )}
              {points.map((p, i) => (
                <circle key={i} cx={p.x} cy={p.y} r={4}
                  fill={color} stroke="#ffffff" strokeWidth={2}
                  style={{ transition: "cx 0.8s ease, cy 0.8s ease" }}
                />
              ))}
            </g>
          );
        })}
      </svg>
      <div className="score-curve-legend">
        {categories.map((cat) => (
          <div key={cat} className="score-curve-legend-item">
            <span className="score-curve-legend-dot" style={{ background: categoryColors[cat] }} />
            <span>{cat}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function ChatBubble({ msg, isNew }: { msg: Message; isNew: boolean }) {
  const isCM = msg.speaker === "CM";
  const isSystem = msg.speaker === "SYSTEM";
  return (
    <div
      className={`chat-bubble ${isSystem ? "chat-system" : isCM ? "chat-cm" : "chat-ccp"} ${isNew ? "chat-new" : ""}`}
    >
      <div className="chat-speaker">
        {isSystem ? "System" : isCM ? "Cardmember" : "CCP"}
      </div>
      <div className="chat-text">{msg.text}</div>
    </div>
  );
}

function QuestionStatusBadge({ status }: { status: ProbingQuestion["status"] }) {
  const statusColor =
    status === "answered" ? "#008000" :
    status === "invalidated" ? "#CF291D" :
    status === "skipped" ? "#53565A" :
    "#F5A623";
  const statusBg =
    status === "answered" ? "#D4EDDA" :
    status === "invalidated" ? "#F8D7DA" :
    status === "skipped" ? "#E2E3E5" :
    "#FFF3CD";
  return (
    <span style={{
      display: "inline-block",
      padding: "1px 8px",
      borderRadius: 10,
      fontSize: 12,
      fontWeight: 700,
      background: statusBg,
      color: statusColor,
      whiteSpace: "nowrap",
      flexShrink: 0,
      marginTop: 2,
      transition: "background 0.8s ease, color 0.8s ease",
    }}>{status}</span>
  );
}

function buildQuestionOrigins(
  activeRun: number,
  copilotResults: CopilotResult[]
): Record<string, number> {
  const origins: Record<string, number> = {};
  for (let r = 0; r <= activeRun && r < copilotResults.length; r++) {
    for (const pq of copilotResults[r].probingQuestions) {
      if (!(pq.text in origins)) {
        origins[pq.text] = r;
      }
    }
  }
  return origins;
}

function QuestionListPanel({
  result,
  prevResult,
  activeRun,
  copilotResults,
}: {
  result: CopilotResult | null;
  prevResult: CopilotResult | null;
  activeRun: number;
  copilotResults: CopilotResult[];
}) {
  if (!result) {
    return (
      <div style={{ padding: 20, color: "#53565A", fontSize: 14, textAlign: "center" }}>
        {"\u23F3"} Waiting for copilot...
      </div>
    );
  }

  const prevStatuses: Record<string, string> = {};
  if (prevResult) {
    prevResult.probingQuestions.forEach((pq) => { prevStatuses[pq.text] = pq.status; });
  }

  const origins = buildQuestionOrigins(activeRun, copilotResults);

  const pending = result.probingQuestions.filter((pq) => pq.status === "pending").length;
  const resolved = result.probingQuestions.filter((pq) => pq.status !== "pending").length;

  const statusBg: Record<string, string> = {
    pending: "#FFF8E1",
    answered: "#D4EDDA",
    invalidated: "#F8D7DA",
    skipped: "#E2E3E5",
  };
  const statusBorder: Record<string, string> = {
    pending: "#F5A62330",
    answered: "#00800030",
    invalidated: "#CF291D30",
    skipped: "#53565A30",
  };

  let lastOriginRun = -1;

  return (
    <div style={{ padding: 16, display: "flex", flexDirection: "column", gap: 12 }}>
      <div style={{ display: "flex", gap: 12, fontSize: 14, color: "#53565A" }}>
        <span style={{ fontWeight: 700 }}>{result.probingQuestions.length} total</span>
        <span style={{ color: "#F5A623" }}>{pending} pending</span>
        <span style={{ color: "#008000" }}>{resolved} resolved</span>
      </div>
      <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
        {result.probingQuestions.map((pq, i) => {
          const isResolved = pq.status !== "pending";
          const prevStatus = prevStatuses[pq.text];
          const isNew = !prevStatus;
          const statusChanged = prevStatus !== undefined && prevStatus !== pq.status;
          const originRun = origins[pq.text] ?? 0;
          const showHeader = originRun !== lastOriginRun;
          if (showHeader) lastOriginRun = originRun;
          const runInfo = copilotResults[originRun];

          return (
            <React.Fragment key={pq.text + i}>
              {showHeader && runInfo && (
                <li style={{
                  fontSize: 14, fontWeight: 700, color: "#006FCF",
                  padding: "10px 4px 6px 4px",
                  marginTop: i > 0 ? 10 : 0,
                  borderBottom: "1px solid #E0E4EA",
                  letterSpacing: 0.3,
                }}>
                  {runInfo.runLabel} &middot; {runInfo.turnRange}
                </li>
              )}
              <li style={{
                display: "flex", alignItems: "flex-start", gap: 8,
                marginBottom: 4, marginTop: 4, padding: "6px 10px",
                borderRadius: 8,
                background: statusBg[pq.status] || "#FFF8E1",
                border: `1px solid ${statusBorder[pq.status] || "#E0E0E0"}`,
                opacity: isResolved ? 0.75 : 1,
                transition: "background 0.8s ease, border-color 0.8s ease, opacity 0.8s ease",
                animation: isNew ? "fadeSlideIn 0.5s ease" : statusChanged ? "statusPulse 0.8s ease" : "none",
              }}>
                <QuestionStatusBadge status={pq.status} />
                <span style={{
                  fontSize: isResolved ? 15 : 16, lineHeight: 1.4,
                  color: isResolved ? "#53565A" : "#00175A",
                  transition: "color 0.8s ease, font-size 0.3s ease",
                }}>
                  {pq.text}
                  {pq.target && (
                    <span style={{
                      color: "#53565A", fontSize: isResolved ? 13 : 14,
                      marginLeft: 4, transition: "font-size 0.3s ease",
                    }}>[{pq.target}]</span>
                  )}
                </span>
              </li>
            </React.Fragment>
          );
        })}
      </ul>
      {result.informationSufficient && (
        <div style={{
          fontSize: 14, fontWeight: 700, color: "#008000",
          padding: "10px 14px", background: "#D4EDDA",
          borderRadius: 8, textAlign: "center",
        }}>
          {"\u2705"} Information sufficient — ready to proceed
        </div>
      )}
    </div>
  );
}

function CaseEligibilityPanel({ result }: { result: CopilotResult }) {
  return (
    <div className="copilot-panel copilot-active">
      <div className="copilot-header">
        <span className="copilot-run-badge">{result.runLabel}</span>
        <span className="copilot-turn-range">{result.turnRange}</span>
      </div>
      <div className="copilot-section">
        <div className="copilot-section-title">Case Opening Eligibility</div>
        <div className="case-eligibility-row">
          <div className={`case-eligibility-card case-eligibility-${result.caseEligibility.fraudCase}`}>
            <div className="case-eligibility-icon">{"\uD83D\uDEE1\uFE0F"}</div>
            <div className="case-eligibility-label">Fraud Case</div>
            <div className="case-eligibility-status">{result.caseEligibility.fraudCase}</div>
          </div>
          <div className={`case-eligibility-card case-eligibility-${result.caseEligibility.disputeCase}`}>
            <div className="case-eligibility-icon">{"\uD83D\uDCCB"}</div>
            <div className="case-eligibility-label">Dispute Case</div>
            <div className="case-eligibility-status">{result.caseEligibility.disputeCase}</div>
          </div>
        </div>
      </div>
    </div>
  );
}

// --- Main Page ---

export default function ReplayPage() {
  const [scenarios, setScenarios] = useState<string[]>([]);
  const [selectedScenario, setSelectedScenario] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Transformed data
  const [messages, setMessages] = useState<Message[]>([]);
  const [copilotResults, setCopilotResults] = useState<CopilotResult[]>([]);
  const [scoreHistory, setScoreHistory] = useState<ScorePoint[]>([]);
  const [caseData, setCaseData] = useState<Record<string, unknown> | null>(null);

  // Playback state
  const [visibleMessages, setVisibleMessages] = useState(0);
  const [activeRun, setActiveRun] = useState(-1);
  const [playing, setPlaying] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const questionEndRef = useRef<HTMLDivElement>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Fetch scenarios on mount
  useEffect(() => {
    fetch("/api/scenarios")
      .then((r) => r.json())
      .then(setScenarios)
      .catch(() => setError("Failed to load scenarios. Is the server running?"));
  }, []);

  // Load scenario data
  const loadScenario = useCallback(async (name: string) => {
    if (!name) return;
    setLoading(true);
    setError("");
    setPlaying(false);
    setVisibleMessages(0);
    setActiveRun(-1);

    try {
      const resp = await fetch(`/api/scenario/${encodeURIComponent(name)}`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data: ScenarioData = await resp.json();

      const { messages: msgs, copilotResults: results, scoreHistory: scores } = transformData(data);
      setMessages(msgs);
      setCopilotResults(results);
      setScoreHistory(scores);
      setCaseData(data.case);
      setPlaying(true);
    } catch (err) {
      setError(`Failed to load scenario: ${err}`);
    } finally {
      setLoading(false);
    }
  }, []);

  // Auto-advance messages
  useEffect(() => {
    if (!playing || visibleMessages >= messages.length) return;
    timerRef.current = setTimeout(() => {
      const nextIdx = visibleMessages;
      setVisibleMessages(nextIdx + 1);
      const msg = messages[nextIdx];
      if (msg.run !== undefined) {
        setActiveRun(msg.run);
      }
    }, 1800);
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [visibleMessages, playing, messages]);

  // Auto-scroll chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [visibleMessages]);

  // Auto-scroll questions
  useEffect(() => {
    questionEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [activeRun]);

  const togglePlay = () => setPlaying((p) => !p);
  const restart = () => {
    setVisibleMessages(0);
    setActiveRun(-1);
    setPlaying(true);
  };

  const allegationType = caseData
    ? String((caseData as Record<string, unknown>).allegation_type || "N/A")
    : "N/A";

  // Scenario not loaded yet — show selector
  if (messages.length === 0 && !loading) {
    return (
      <div className="demo-page">
        <div className="demo-header">
          <div className="demo-title">
            CCP Desktop Replay
            <span className="demo-subtitle">Load a simulation scenario</span>
          </div>
          <div className="demo-controls">
            <a href="#/" className="demo-btn-link">Back to Workflow</a>
          </div>
        </div>
        <div style={{
          display: "flex", flexDirection: "column", alignItems: "center",
          justifyContent: "center", flex: 1, gap: 16, padding: 40,
        }}>
          {error && (
            <div style={{ color: "#CF291D", fontSize: 14, marginBottom: 8 }}>{error}</div>
          )}
          <select
            value={selectedScenario}
            onChange={(e) => setSelectedScenario(e.target.value)}
            style={{
              padding: "10px 16px", fontSize: 16, borderRadius: 8,
              border: "2px solid #006FCF", minWidth: 300,
            }}
          >
            <option value="">Select a scenario...</option>
            {scenarios.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
          <button
            onClick={() => loadScenario(selectedScenario)}
            disabled={!selectedScenario}
            style={{
              padding: "10px 32px", fontSize: 16, fontWeight: 700,
              borderRadius: 8, border: "none", cursor: "pointer",
              background: selectedScenario ? "#006FCF" : "#C8C9C7",
              color: "#fff",
            }}
          >
            Load & Replay
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="demo-page">
      {/* Header */}
      <div className="demo-header">
        <div className="demo-title">
          CCP Desktop Replay
          <span className="demo-subtitle">{selectedScenario}</span>
        </div>
        <div className="demo-controls">
          <button className="demo-btn" onClick={restart} title="Restart">
            {"\u23EE"}
          </button>
          <button className="demo-btn" onClick={togglePlay} title={playing ? "Pause" : "Play"}>
            {playing ? "\u23F8" : "\u25B6"}
          </button>
          <button
            className="demo-btn"
            onClick={() => {
              setMessages([]);
              setCopilotResults([]);
              setScoreHistory([]);
              setCaseData(null);
              setPlaying(false);
              setVisibleMessages(0);
              setActiveRun(-1);
            }}
            title="Change Scenario"
          >
            {"\u21A9"}
          </button>
          <a href="#/" className="demo-btn-link">Back to Workflow</a>
        </div>
      </div>

      {/* Scenario info bar */}
      <div style={{
        padding: "8px 24px",
        borderBottom: "1px solid #E0E0E0",
        background: "rgba(0, 111, 207, 0.04)",
        display: "flex", alignItems: "center", gap: 10, flexShrink: 0,
      }}>
        <span style={{
          fontSize: 15, fontWeight: 700, textTransform: "uppercase" as const,
          letterSpacing: 1.2, color: "#53565A", flexShrink: 0,
        }}>Replay</span>
        <span style={{ width: 1, height: 20, background: "#C8C9C7", flexShrink: 0 }} />
        <span style={{ fontSize: 15, color: "#006FCF", fontWeight: 600 }}>
          Allegation: {allegationType}
        </span>
        <span style={{ width: 1, height: 20, background: "#C8C9C7", flexShrink: 0 }} />
        <span style={{ fontSize: 14, color: "#53565A" }}>
          Turn {visibleMessages}/{messages.length} | Run {activeRun + 1}/{copilotResults.length}
        </span>
      </div>

      {/* Main content — three panels */}
      <div className="demo-content">
        {/* Left: Conversation */}
        <div className="demo-chat-panel">
          <div className="demo-panel-header">
            <span className="demo-panel-dot" style={{ background: "#008000" }} />
            Live Conversation
          </div>
          <div className="demo-chat-scroll">
            {messages.slice(0, visibleMessages).map((msg, i) => (
              <React.Fragment key={i}>
                {msg.run !== undefined && (
                  <div className="chat-run-divider">
                    <span>Copilot Run {msg.run + 1} triggered</span>
                  </div>
                )}
                <ChatBubble msg={msg} isNew={i === visibleMessages - 1} />
              </React.Fragment>
            ))}
            <div ref={chatEndRef} />
          </div>
        </div>

        {/* Middle: Question List */}
        <div className="demo-question-panel">
          <div className="demo-panel-header">
            <span className="demo-panel-dot" style={{ background: "#F5A623" }} />
            Probing Questions
          </div>
          <div className="demo-question-scroll">
            <QuestionListPanel
              result={activeRun >= 0 ? copilotResults[activeRun] : null}
              prevResult={activeRun > 0 ? copilotResults[activeRun - 1] : null}
              activeRun={activeRun}
              copilotResults={copilotResults}
            />
            <div ref={questionEndRef} />
          </div>
        </div>

        {/* Right: Copilot Run Results */}
        <div className="demo-copilot-panel">
          <div className="demo-panel-header">
            <span className="demo-panel-dot" style={{ background: "#006FCF" }} />
            Copilot Run Results
          </div>
          <div className="demo-copilot-scroll">
            {activeRun < 0 ? (
              <div className="copilot-waiting">
                <div className="copilot-waiting-icon">{"\u23F3"}</div>
                <div>Waiting for copilot to trigger...</div>
                <div className="copilot-waiting-sub">Conversation in progress</div>
              </div>
            ) : (
              <>
                <ScoreCurveChart visibleRuns={activeRun + 1} scoreHistory={scoreHistory} />
                {/* <CaseEligibilityPanel result={copilotResults[activeRun]} /> */}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
