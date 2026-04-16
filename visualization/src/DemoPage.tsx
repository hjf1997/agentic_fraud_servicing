import React, { useState, useEffect, useRef } from "react";

// --- Types ---

type Message = {
  speaker: "CM" | "CCP";
  text: string;
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
  caseEligibility: { fraudCase: "eligible" | "blocked"; disputeCase: "eligible" | "blocked" };
};

// --- Simulated Data ---

const conversation: (Message & { run?: number })[] = [
  // Pre-copilot
  { speaker: "CCP", text: "Thank you for calling American Express. How can I help you today?" },
  { speaker: "CM", text: "Hi, I need to report a fraudulent charge on my card. There's a $1,850 transaction from LuxFurniture that I never made." },
  { speaker: "CCP", text: "I'm sorry to hear that. Let me pull up your account. Can you confirm the last four digits of your card?" },
  { speaker: "CM", text: "It ends in 7293." },
  { speaker: "CCP", text: "I can see the charge \u2014 LuxFurniture, March 20th, $1,850.00. You're saying you didn't authorize this transaction?" },
  // Run 1
  { speaker: "CM", text: "No, absolutely not. I don't know what LuxFurniture is. I've never shopped there.", run: 1 },
  { speaker: "CCP", text: "I understand. Was your card lost or stolen at any point around that date?" },
  { speaker: "CM", text: "No, I've had the card with me the whole time." },
  // Run 2
  { speaker: "CCP", text: "Have you shared your card details with anyone, or noticed any other unfamiliar charges?", run: 2 },
  { speaker: "CM", text: "No, I never share my card. This is the only charge I don't recognize." },
  { speaker: "CCP", text: "Do you have any family members or authorized users on this account?" },
  // Run 3
  { speaker: "CM", text: "No, it's just me on the account. Nobody else has access.", run: 3 },
  { speaker: "CCP", text: "Our records show this was an online transaction with your billing address. Did you receive any suspicious emails asking you to click a link or verify your account?" },
  { speaker: "CM", text: "No, nothing like that. I'm very careful with emails." },
  // Run 4
  { speaker: "CCP", text: "I see. The transaction used your saved card credentials and was shipped to your billing address. Can you confirm your current address for me?", run: 4 },
  { speaker: "CM", text: "It's 142 Oak Street, apartment 5B. But I'm telling you, I didn't order anything." },
  { speaker: "CCP", text: "That matches the shipping address on the order. Did anyone else have access to your home to receive a delivery around March 22nd?" },
  // Run 5
  { speaker: "CM", text: "No, I live alone. Maybe someone stole the package after delivery? I really don't know what this charge is.", run: 5 },
  { speaker: "CCP", text: "I understand. Let me check a few more things on our end." },
  { speaker: "CCP", text: "I can see the device used for this purchase was an iPhone registered to your account. Does that sound familiar?" },
  // Run 6
  { speaker: "CM", text: "I mean, I have an iPhone, but I didn't use it to buy anything from that store. Maybe my phone was hacked?", run: 6 },
  { speaker: "CCP", text: "That's possible. We'll look into that. Were you using any public Wi-Fi around that time?" },
  { speaker: "CM", text: "I sometimes use the coffee shop Wi-Fi, but I wouldn't buy furniture on public Wi-Fi." },
  // Run 7
  { speaker: "CCP", text: "I appreciate your patience. I want to be thorough here. Our system shows a prior case on your account \u2014 did you previously contact us about this same LuxFurniture charge?", run: 7 },
  { speaker: "CM", text: "Uh... what do you mean?" },
  { speaker: "CCP", text: "Our records show you called on March 25th and opened a dispute case for this same $1,850 charge from LuxFurniture. You told our colleague that you ordered a dining table but received a damaged item, and you wanted a chargeback because the merchant refused a return." },
  // Run 8
  { speaker: "CM", text: "Oh... yes, I did call about that. But the situation has changed. I realized the charge might actually be fraud because I don't think I actually placed that order myself.", run: 8 },
  { speaker: "CCP", text: "I see. But in the previous call you confirmed you placed the order, described the item you received, and provided details about your communication with the merchant about the return. Those details suggest you were aware of the purchase." },
  { speaker: "CM", text: "I was confused at the time. I think someone may have used my card and I just assumed it was my order." },
  // Run 9
  { speaker: "CCP", text: "I understand this can be confusing. However, the previous dispute case includes your confirmation that you authorized the purchase and received merchandise. Reporting the same charge as unauthorized fraud after filing a merchant dispute creates a significant inconsistency.", run: 9 },
  { speaker: "CM", text: "I see. So what happens now? I just want the charge resolved." },
  { speaker: "CCP", text: "Based on the full history, this case is consistent with a merchant dispute rather than unauthorized fraud. The existing dispute case is already being investigated by our team." },
  // Run 10
  { speaker: "CM", text: "OK, so the original dispute is still being handled?", run: 10 },
  { speaker: "CCP", text: "Yes, your dispute case DSP-2024-39102 is still active. Our investigations team is reviewing the merchant's response. You should hear back within 10 business days. We cannot open a separate fraud case for a charge you previously confirmed authorizing." },
  { speaker: "CM", text: "Alright, I understand. I'll wait for the dispute outcome then." },
  { speaker: "CCP", text: "That's the best path forward. If you have any new information about the merchant or the damaged item, you can call us to add it to the existing case. Is there anything else I can help with?" },
  { speaker: "CM", text: "No, that's it. Thank you." },
];

const copilotResults: CopilotResult[] = [
  {
    runLabel: "Copilot Run 1", turnRange: "Turn 1-8",
    probingQuestions: [
      { text: "Ask if card was lost, stolen, or shared with anyone", status: "pending", target: "FRAUD" },
      { text: "Check if there are other unrecognized charges", status: "pending", target: "FRAUD" },
    ],
    informationSufficient: false,
    caseEligibility: { fraudCase: "blocked", disputeCase: "blocked" },
  },
  {
    runLabel: "Copilot Run 2", turnRange: "Turn 9-11",
    probingQuestions: [
      { text: "Ask if card was lost, stolen, or shared with anyone", status: "answered", target: "FRAUD" },
      { text: "Check if there are other unrecognized charges", status: "answered", target: "FRAUD" },
    ],
    informationSufficient: false,
    caseEligibility: { fraudCase: "blocked", disputeCase: "blocked" },
  },
  {
    runLabel: "Copilot Run 3", turnRange: "Turn 12-14",
    probingQuestions: [
      { text: "Ask if card was lost, stolen, or shared with anyone", status: "answered", target: "FRAUD" },
      { text: "Check if there are other unrecognized charges", status: "answered", target: "FRAUD" },
      { text: "Ask if CM received phishing emails or suspicious links", status: "pending", target: "SCAM" },
    ],
    informationSufficient: false,
    caseEligibility: { fraudCase: "blocked", disputeCase: "blocked" },
  },
  {
    runLabel: "Copilot Run 4", turnRange: "Turn 15-17",
    probingQuestions: [
      { text: "Ask if card was lost, stolen, or shared with anyone", status: "answered", target: "FRAUD" },
      { text: "Check if there are other unrecognized charges", status: "answered", target: "FRAUD" },
      { text: "Ask if CM received phishing emails or suspicious links", status: "answered", target: "SCAM" },
    ],
    informationSufficient: false,
    caseEligibility: { fraudCase: "eligible", disputeCase: "blocked" },
  },
  {
    runLabel: "Copilot Run 5", turnRange: "Turn 18-20",
    probingQuestions: [
      { text: "Ask if card was lost, stolen, or shared with anyone", status: "answered", target: "FRAUD" },
      { text: "Check if there are other unrecognized charges", status: "answered", target: "FRAUD" },
      { text: "Ask if CM received phishing emails or suspicious links", status: "answered", target: "SCAM" },
      { text: "Confirm whether anyone else could have received the delivery", status: "pending", target: "BOGUS" },
    ],
    informationSufficient: false,
    caseEligibility: { fraudCase: "eligible", disputeCase: "blocked" },
  },
  {
    runLabel: "Copilot Run 6", turnRange: "Turn 21-23",
    probingQuestions: [
      { text: "Ask if card was lost, stolen, or shared with anyone", status: "answered", target: "FRAUD" },
      { text: "Check if there are other unrecognized charges", status: "answered", target: "FRAUD" },
      { text: "Ask if CM received phishing emails or suspicious links", status: "invalidated", target: "SCAM" },
      { text: "Confirm whether anyone else could have received the delivery", status: "answered", target: "BOGUS" },
    ],
    informationSufficient: false,
    caseEligibility: { fraudCase: "eligible", disputeCase: "blocked" },
  },
  {
    runLabel: "Copilot Run 7", turnRange: "Turn 24-26",
    probingQuestions: [
      { text: "Ask if card was lost, stolen, or shared with anyone", status: "answered", target: "FRAUD" },
      { text: "Check if there are other unrecognized charges", status: "answered", target: "FRAUD" },
      { text: "Ask if CM received phishing emails or suspicious links", status: "invalidated", target: "SCAM" },
      { text: "Confirm whether anyone else could have received the delivery", status: "answered", target: "BOGUS" },
      { text: "Ask if CM previously filed a dispute for this same charge", status: "pending", target: "BOGUS" },
    ],
    informationSufficient: false,
    caseEligibility: { fraudCase: "eligible", disputeCase: "blocked" },
  },
  {
    runLabel: "Copilot Run 8", turnRange: "Turn 27-29",
    probingQuestions: [
      { text: "Ask if card was lost, stolen, or shared with anyone", status: "answered", target: "FRAUD" },
      { text: "Check if there are other unrecognized charges", status: "answered", target: "FRAUD" },
      { text: "Ask if CM received phishing emails or suspicious links", status: "invalidated", target: "SCAM" },
      { text: "Confirm whether anyone else could have received the delivery", status: "skipped", target: "BOGUS" },
      { text: "Ask if CM previously filed a dispute for this same charge", status: "answered", target: "BOGUS" },
      { text: "Clarify why allegation changed from dispute to fraud", status: "pending", target: "BOGUS" },
    ],
    informationSufficient: false,
    caseEligibility: { fraudCase: "blocked", disputeCase: "blocked" },
  },
  {
    runLabel: "Copilot Run 9", turnRange: "Turn 30-32",
    probingQuestions: [
      { text: "Ask if card was lost, stolen, or shared with anyone", status: "answered", target: "FRAUD" },
      { text: "Check if there are other unrecognized charges", status: "answered", target: "FRAUD" },
      { text: "Ask if CM received phishing emails or suspicious links", status: "invalidated", target: "SCAM" },
      { text: "Confirm whether anyone else could have received the delivery", status: "skipped", target: "BOGUS" },
      { text: "Ask if CM previously filed a dispute for this same charge", status: "answered", target: "BOGUS" },
      { text: "Clarify why allegation changed from dispute to fraud", status: "answered", target: "BOGUS" },
    ],
    informationSufficient: true,
    caseEligibility: { fraudCase: "blocked", disputeCase: "blocked" },
  },
  {
    runLabel: "Copilot Run 10", turnRange: "Turn 33-37",
    probingQuestions: [
      { text: "Ask if card was lost, stolen, or shared with anyone", status: "answered", target: "FRAUD" },
      { text: "Check if there are other unrecognized charges", status: "answered", target: "FRAUD" },
      { text: "Ask if CM received phishing emails or suspicious links", status: "invalidated", target: "SCAM" },
      { text: "Confirm whether anyone else could have received the delivery", status: "skipped", target: "BOGUS" },
      { text: "Ask if CM previously filed a dispute for this same charge", status: "answered", target: "BOGUS" },
      { text: "Clarify why allegation changed from dispute to fraud", status: "answered", target: "BOGUS" },
    ],
    informationSufficient: true,
    caseEligibility: { fraudCase: "blocked", disputeCase: "blocked" },
  },
];

// --- Score history for the curve chart ---

type ScorePoint = { label: string; scores: Record<string, number> };

const scoreHistory: ScorePoint[] = [
  // Prior: uniform
  { label: "Prior", scores: { "Fraud": 0.25, "Bogus": 0.25, "Scam": 0.25, "Billing Dispute": 0.25 } },
  // R1: CM claims unauthorized charge — third-party fraud signal
  { label: "R1", scores: { "Fraud": 0.60, "Bogus": 0.10, "Scam": 0.10, "Billing Dispute": 0.20 } },
  // R2: card not lost, no sharing, single charge — still looks like fraud
  { label: "R2", scores: { "Fraud": 0.62, "Bogus": 0.12, "Scam": 0.08, "Billing Dispute": 0.18 } },
  // R3: no phishing, no suspicious links — scam unlikely, but device/address match raises doubt
  { label: "R3", scores: { "Fraud": 0.55, "Bogus": 0.18, "Scam": 0.05, "Billing Dispute": 0.22 } },
  // R4: shipped to CM's address, used saved credentials — contradiction with fraud claim
  { label: "R4", scores: { "Fraud": 0.42, "Bogus": 0.28, "Scam": 0.04, "Billing Dispute": 0.26 } },
  // R5: CM's device used, lives alone, no one else could receive — first-party fraud rises
  { label: "R5", scores: { "Fraud": 0.32, "Bogus": 0.35, "Scam": 0.03, "Billing Dispute": 0.30 } },
  // R6: CM's own iPhone confirmed, weak hacking explanation
  { label: "R6", scores: { "Fraud": 0.25, "Bogus": 0.40, "Scam": 0.03, "Billing Dispute": 0.32 } },
  // R7: TURNING POINT — prior dispute case discovered, CM evasive
  { label: "R7", scores: { "Fraud": 0.10, "Bogus": 0.35, "Scam": 0.03, "Billing Dispute": 0.52 } },
  // R8: CM admits prior dispute, confirms authorized purchase — fraud claim collapses
  { label: "R8", scores: { "Fraud": 0.05, "Bogus": 0.30, "Scam": 0.02, "Billing Dispute": 0.63 } },
  // R9: inconsistency confirmed, dispute is the true category
  { label: "R9", scores: { "Fraud": 0.04, "Bogus": 0.25, "Scam": 0.02, "Billing Dispute": 0.69 } },
  // R10: final — dispute dominant, first-party fraud elevated, fraud collapsed
  { label: "R10", scores: { "Fraud": 0.03, "Bogus": 0.22, "Scam": 0.02, "Billing Dispute": 0.73 } },
];

const categoryColors: Record<string, string> = {
  "Fraud": "#006FCF",
  "Bogus": "#F5A623",
  "Scam": "#CF291D",
  "Billing Dispute": "#008000",
};

const categories = Object.keys(categoryColors);

// --- Components ---

function ScoreCurveChart({ visibleRuns }: { visibleRuns: number }) {
  // visibleRuns: 0 = prior only, 1 = prior + run1, 2 = all three
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
        {/* Grid lines */}
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

        {/* X-axis labels */}
        {scoreHistory.map((p, i) => (
          <text
            key={i} x={toX(i)} y={H - 6}
            textAnchor="middle" fill={i < pointCount ? "#53565A" : "#C8C9C7"} fontSize={11} fontWeight={600}
          >
            {p.label}
          </text>
        ))}

        {/* Lines and dots for each category */}
        {categories.map((cat) => {
          const color = categoryColors[cat];
          const points = visibleData.map((p, i) => ({ x: toX(i), y: toY(p.scores[cat]) }));

          // Build path
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

      {/* Legend */}
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
  return (
    <div
      className={`chat-bubble ${isCM ? "chat-cm" : "chat-ccp"} ${isNew ? "chat-new" : ""}`}
    >
      <div className="chat-speaker">{isCM ? "Cardmember" : "CCP"}</div>
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

/**
 * Determine which copilot run first introduced each question by scanning all
 * results up to and including the active run.
 */
function buildQuestionOrigins(activeRun: number): Record<string, number> {
  const origins: Record<string, number> = {};
  for (let r = 0; r <= activeRun; r++) {
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
}: {
  result: CopilotResult | null;
  prevResult: CopilotResult | null;
  activeRun: number;
}) {
  if (!result) {
    return (
      <div style={{ padding: 20, color: "#53565A", fontSize: 14, textAlign: "center" }}>
        {"\u23F3"} Waiting for copilot...
      </div>
    );
  }

  // Build a map of previous statuses by question text to detect changes
  const prevStatuses: Record<string, string> = {};
  if (prevResult) {
    prevResult.probingQuestions.forEach((pq) => { prevStatuses[pq.text] = pq.status; });
  }

  // Determine which run each question was first generated in
  const origins = buildQuestionOrigins(activeRun);

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

  // Group questions by their origin run for section headers
  let lastOriginRun = -1;

  return (
    <div style={{ padding: 16, display: "flex", flexDirection: "column", gap: 12 }}>
      {/* Summary counts */}
      <div style={{ display: "flex", gap: 12, fontSize: 14, color: "#53565A" }}>
        <span style={{ fontWeight: 700 }}>{result.probingQuestions.length} total</span>
        <span style={{ color: "#F5A623" }}>{pending} pending</span>
        <span style={{ color: "#008000" }}>{resolved} resolved</span>
      </div>

      {/* All questions in original order, with run section headers */}
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
            <React.Fragment key={pq.text}>
              {showHeader && (
                <li style={{
                  fontSize: 14,
                  fontWeight: 700,
                  color: "#006FCF",
                  padding: "10px 4px 6px 4px",
                  marginTop: i > 0 ? 10 : 0,
                  borderBottom: "1px solid #E0E4EA",
                  letterSpacing: 0.3,
                }}>
                  {runInfo.runLabel} &middot; {runInfo.turnRange}
                </li>
              )}
              <li style={{
                display: "flex",
                alignItems: "flex-start",
                gap: 8,
                marginBottom: 4,
                marginTop: 4,
                padding: "6px 10px",
                borderRadius: 8,
                background: statusBg[pq.status] || "#FFF8E1",
                border: `1px solid ${statusBorder[pq.status] || "#E0E0E0"}`,
                opacity: isResolved ? 0.75 : 1,
                transition: "background 0.8s ease, border-color 0.8s ease, opacity 0.8s ease",
                animation: isNew ? "fadeSlideIn 0.5s ease" : statusChanged ? "statusPulse 0.8s ease" : "none",
              }}>
                <QuestionStatusBadge status={pq.status} />
                <span style={{
                  fontSize: isResolved ? 15 : 16,
                  lineHeight: 1.4,
                  color: isResolved ? "#53565A" : "#00175A",
                  fontWeight: pq.text.includes("previously filed a dispute") ? 700 : "normal",
                  transition: "color 0.8s ease, font-size 0.3s ease",
                }}>
                  {pq.text}
                  <span style={{
                    color: "#53565A",
                    fontSize: isResolved ? 13 : 14,
                    marginLeft: 4,
                    transition: "font-size 0.3s ease",
                  }}>[{pq.target}]</span>
                </span>
              </li>
            </React.Fragment>
          );
        })}
      </ul>

      {/* Information sufficient banner */}
      {result.informationSufficient && (
        <div style={{
          fontSize: 14,
          fontWeight: 700,
          color: "#008000",
          padding: "10px 14px",
          background: "#D4EDDA",
          borderRadius: 8,
          textAlign: "center",
        }}>
          {"\u2705"} Information sufficient — ready to proceed
        </div>
      )}
    </div>
  );
}

function CaseEligibilityPanel({ result, isActive }: { result: CopilotResult; isActive: boolean }) {
  return (
    <div className={`copilot-panel ${isActive ? "copilot-active" : ""}`}>
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

export default function DemoPage() {
  const [visibleMessages, setVisibleMessages] = useState(0);
  const [activeRun, setActiveRun] = useState(-1); // -1 = no run yet
  const [playing, setPlaying] = useState(true);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const questionEndRef = useRef<HTMLDivElement>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Auto-advance messages
  useEffect(() => {
    if (!playing || visibleMessages >= conversation.length) return;
    timerRef.current = setTimeout(() => {
      const nextIdx = visibleMessages;
      setVisibleMessages(nextIdx + 1);
      // Check if this message triggers a copilot run
      const msg = conversation[nextIdx];
      if (msg.run !== undefined) {
        setActiveRun(msg.run - 1); // 0-indexed
      }
    }, 1800);
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [visibleMessages, playing]);

  // Auto-scroll chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [visibleMessages]);

  // Auto-scroll probing questions
  useEffect(() => {
    questionEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [activeRun]);

  const togglePlay = () => setPlaying((p) => !p);
  const restart = () => {
    setVisibleMessages(0);
    setActiveRun(-1);
    setPlaying(true);
  };

  return (
    <div className="demo-page">
      {/* Header */}
      <div className="demo-header">
        <div className="demo-title">
          CCP Desktop
          <span className="demo-subtitle">Fraud Servicing Call Simulation</span>
        </div>
        <div className="demo-controls">
          <button className="demo-btn" onClick={restart} title="Restart">
            {"\u23EE"}
          </button>
          <button className="demo-btn" onClick={togglePlay} title={playing ? "Pause" : "Play"}>
            {playing ? "\u23F8" : "\u25B6"}
          </button>
          <a href="#/" className="demo-btn-link">Back to Workflow</a>
        </div>
      </div>

      {/* Scenario summary bar */}
      <div style={{
        padding: "8px 24px",
        borderBottom: "1px solid #E0E0E0",
        background: activeRun >= 6 ? "rgba(207, 41, 29, 0.04)" : "rgba(0, 111, 207, 0.04)",
        transition: "background 1s ease",
        display: "flex",
        alignItems: "center",
        gap: 10,
        flexShrink: 0,
      }}>
        <span style={{
          fontSize: 15, fontWeight: 700, textTransform: "uppercase" as const,
          letterSpacing: 1.2, color: "#53565A", flexShrink: 0,
        }}>Simulation Narrator</span>
        <span style={{
          width: 1, height: 20, background: "#C8C9C7", flexShrink: 0,
        }} />
        <span style={{
          width: 10, height: 10, borderRadius: "50%", flexShrink: 0,
          background: activeRun >= 6 ? "#CF291D" : "#006FCF",
          transition: "background 0.8s ease",
        }} />
        <span style={{
          fontSize: 17, fontWeight: 600, letterSpacing: 0.3,
          color: activeRun >= 6 ? "#CF291D" : "#006FCF",
          transition: "color 0.8s ease",
        }}>
          {activeRun < 0
            ? "Scenario: CM claims fraud, but previously opened a dispute case for the same charge"
            : activeRun < 6
            ? "Based on CM's narrative, fraud likelihood increases and fraud case is eligible to open"
            : activeRun === 6
            ? "Copilot fetched previous dispute record and raised the question: did you claim this case as a dispute?"
            : "CM admits prior dispute claim \u2014 copilot shifts to dispute; fraud case opening is blocked"}
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
            {conversation.slice(0, visibleMessages).map((msg, i) => (
              <React.Fragment key={i}>
                {msg.run !== undefined && (
                  <div className="chat-run-divider">
                    <span>Copilot Run {msg.run} triggered</span>
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
                <ScoreCurveChart visibleRuns={activeRun + 1} />
                {/* <CaseEligibilityPanel result={copilotResults[activeRun]} isActive={true} /> */}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
