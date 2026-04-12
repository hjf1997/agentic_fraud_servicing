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
  { speaker: "CM", text: "Hi, I noticed a charge on my card for $487.50 from an electronics store. I never made this purchase." },
  { speaker: "CCP", text: "I'm sorry to hear that. Let me pull up your account. Can you confirm the last four digits of your card?" },
  { speaker: "CM", text: "It ends in 4821." },
  { speaker: "CCP", text: "I can see the transaction from TechMart on March 15th for $487.50. You're saying you didn't authorize this?" },
  // Run 1
  { speaker: "CM", text: "No, I've never even heard of that store. I had my card with me the whole time.", run: 1 },
  { speaker: "CCP", text: "I understand. Were you traveling or did anyone else have access to your card recently?" },
  { speaker: "CM", text: "No, I was home all week. Nobody else uses my card." },
  // Run 2
  { speaker: "CCP", text: "Have you noticed any other unfamiliar charges on your account?", run: 2 },
  { speaker: "CM", text: "Actually yes, there's another one for $52.99 from a streaming service I don't use." },
  { speaker: "CCP", text: "Did you receive any emails or texts asking you to verify your card information recently?" },
  // Run 3
  { speaker: "CM", text: "Now that you mention it, I got a text last week saying my account was locked and to click a link to verify. I clicked it and entered my card details.", run: 3 },
  { speaker: "CCP", text: "That sounds like a phishing attempt. Can you tell me more about that text \u2014 what number it came from?" },
  // Run 4
  { speaker: "CM", text: "It was a short code, something like 55247. The website looked exactly like the Amex login page.", run: 4 },
  { speaker: "CCP", text: "I see. And after you entered your details on that site, when did the unauthorized charges start appearing?" },
  { speaker: "CM", text: "The TechMart charge showed up two days after I clicked that link. The streaming one appeared the next day." },
  // Run 5
  { speaker: "CCP", text: "That timeline is very consistent with credential theft. Have you changed your password since then?", run: 5 },
  { speaker: "CM", text: "No, I didn't know I needed to. Should I?" },
  { speaker: "CCP", text: "Yes, absolutely. We'll also need to secure your account. Did you use the same password for any other financial services?" },
  // Run 6
  { speaker: "CM", text: "I... I think I use the same password for my bank and a couple of shopping sites.", run: 6 },
  { speaker: "CCP", text: "I strongly recommend changing those as well. Now, I want to confirm \u2014 you definitely did not make the $487.50 purchase at TechMart?" },
  { speaker: "CM", text: "Absolutely not. I don't even know where that store is." },
  // Run 7
  { speaker: "CCP", text: "Our records show the TechMart transaction was an in-store chip purchase in Dallas, Texas. Can you confirm your location on March 15th?", run: 7 },
  { speaker: "CM", text: "I was in New York. I haven't been to Texas in years. You can check my other transactions \u2014 I was using my card at restaurants in Manhattan that same day." },
  // Run 8
  { speaker: "CCP", text: "That's very helpful. I can see Manhattan transactions on your account that same day, which confirms you couldn't have been in Dallas.", run: 8 },
  { speaker: "CM", text: "So someone cloned my card after I entered my details on that fake website?" },
  { speaker: "CCP", text: "That appears to be the case. The phishing site likely captured your full card details, which were then used to create a counterfeit card." },
  // Run 9
  { speaker: "CM", text: "This is really scary. What happens next? Will I get my money back?", run: 9 },
  { speaker: "CCP", text: "Yes, I'm going to file a fraud claim and a scam protection claim for both transactions. You'll receive provisional credit within 24-48 hours." },
  { speaker: "CCP", text: "We'll also issue you a new card number immediately. The old card ending in 4821 will be deactivated." },
  // Run 10
  { speaker: "CM", text: "OK, thank you. Is there anything else I need to do?", run: 10 },
  { speaker: "CCP", text: "Please change your passwords for all financial sites, enable two-factor authentication, and monitor your credit report. I'm also flagging the phishing number 55247 to our security team." },
  { speaker: "CM", text: "I will. Thank you so much for your help." },
  { speaker: "CCP", text: "You're welcome. Your new card will arrive in 3-5 business days. Your case reference number is FRD-2024-84721. Is there anything else I can help with?" },
  { speaker: "CM", text: "No, that's everything. Thank you again." },
];

const copilotResults: CopilotResult[] = [
  {
    runLabel: "Copilot Run 1", turnRange: "Turn 1-8",
    probingQuestions: [
      { text: "Ask if CM received suspicious emails or texts recently", status: "pending", target: "SCAM" },
      { text: "Check if there are additional unauthorized charges", status: "pending", target: "THIRD_PARTY_FRAUD" },
      { text: "Confirm whether card has chip or contactless enabled", status: "pending", target: "THIRD_PARTY_FRAUD" },
    ],
    informationSufficient: false,
    caseEligibility: { fraudCase: "blocked", disputeCase: "blocked" },
  },
  {
    runLabel: "Copilot Run 2", turnRange: "Turn 9-11",
    probingQuestions: [
      { text: "Ask if CM received suspicious emails or texts recently", status: "pending", target: "SCAM" },
      { text: "Check if there are additional unauthorized charges", status: "answered", target: "THIRD_PARTY_FRAUD" },
      { text: "Confirm whether card has chip or contactless enabled", status: "pending", target: "THIRD_PARTY_FRAUD" },
      { text: "Probe for social engineering indicators", status: "pending", target: "SCAM" },
      { text: "Verify the $52.99 streaming charge details", status: "pending", target: "THIRD_PARTY_FRAUD" },
    ],
    informationSufficient: false,
    caseEligibility: { fraudCase: "blocked", disputeCase: "blocked" },
  },
  {
    runLabel: "Copilot Run 3", turnRange: "Turn 12-13",
    probingQuestions: [
      { text: "Ask if CM received suspicious emails or texts recently", status: "answered", target: "SCAM" },
      { text: "Check if there are additional unauthorized charges", status: "answered", target: "THIRD_PARTY_FRAUD" },
      { text: "Confirm whether card has chip or contactless enabled", status: "pending", target: "THIRD_PARTY_FRAUD" },
      { text: "Probe for social engineering indicators", status: "answered", target: "SCAM" },
      { text: "Verify the $52.99 streaming charge details", status: "pending", target: "THIRD_PARTY_FRAUD" },
      { text: "Collect phishing text details: sender number, URL domain", status: "pending", target: "SCAM" },
      { text: "Ask when CM entered credentials on the site", status: "pending", target: "SCAM" },
    ],
    informationSufficient: false,
    caseEligibility: { fraudCase: "blocked", disputeCase: "blocked" },
  },
  {
    runLabel: "Copilot Run 4", turnRange: "Turn 14-16",
    probingQuestions: [
      { text: "Ask if CM received suspicious emails or texts recently", status: "answered", target: "SCAM" },
      { text: "Check if there are additional unauthorized charges", status: "answered", target: "THIRD_PARTY_FRAUD" },
      { text: "Confirm whether card has chip or contactless enabled", status: "skipped", target: "THIRD_PARTY_FRAUD" },
      { text: "Probe for social engineering indicators", status: "answered", target: "SCAM" },
      { text: "Verify the $52.99 streaming charge details", status: "invalidated", target: "THIRD_PARTY_FRAUD" },
      { text: "Collect phishing text details: sender number, URL domain", status: "answered", target: "SCAM" },
      { text: "Ask when CM entered credentials on the site", status: "answered", target: "SCAM" },
      { text: "Check for additional compromised accounts", status: "pending", target: "SCAM" },
    ],
    informationSufficient: false,
    caseEligibility: { fraudCase: "eligible", disputeCase: "blocked" },
  },
  {
    runLabel: "Copilot Run 5", turnRange: "Turn 17-19",
    probingQuestions: [
      { text: "Ask if CM received suspicious emails or texts recently", status: "answered", target: "SCAM" },
      { text: "Check if there are additional unauthorized charges", status: "answered", target: "THIRD_PARTY_FRAUD" },
      { text: "Confirm whether card has chip or contactless enabled", status: "skipped", target: "THIRD_PARTY_FRAUD" },
      { text: "Probe for social engineering indicators", status: "answered", target: "SCAM" },
      { text: "Verify the $52.99 streaming charge details", status: "invalidated", target: "THIRD_PARTY_FRAUD" },
      { text: "Collect phishing text details: sender number, URL domain", status: "answered", target: "SCAM" },
      { text: "Ask when CM entered credentials on the site", status: "answered", target: "SCAM" },
      { text: "Check for additional compromised accounts", status: "answered", target: "SCAM" },
      { text: "Ask about password reuse across services", status: "pending", target: "SCAM" },
    ],
    informationSufficient: false,
    caseEligibility: { fraudCase: "eligible", disputeCase: "blocked" },
  },
  {
    runLabel: "Copilot Run 6", turnRange: "Turn 20-22",
    probingQuestions: [
      { text: "Ask if CM received suspicious emails or texts recently", status: "answered", target: "SCAM" },
      { text: "Check if there are additional unauthorized charges", status: "answered", target: "THIRD_PARTY_FRAUD" },
      { text: "Confirm whether card has chip or contactless enabled", status: "skipped", target: "THIRD_PARTY_FRAUD" },
      { text: "Probe for social engineering indicators", status: "answered", target: "SCAM" },
      { text: "Verify the $52.99 streaming charge details", status: "invalidated", target: "THIRD_PARTY_FRAUD" },
      { text: "Collect phishing text details: sender number, URL domain", status: "answered", target: "SCAM" },
      { text: "Ask when CM entered credentials on the site", status: "answered", target: "SCAM" },
      { text: "Check for additional compromised accounts", status: "answered", target: "SCAM" },
      { text: "Ask about password reuse across services", status: "answered", target: "SCAM" },
      { text: "Verify CM's location on date of in-store transaction", status: "pending", target: "THIRD_PARTY_FRAUD" },
    ],
    informationSufficient: false,
    caseEligibility: { fraudCase: "eligible", disputeCase: "blocked" },
  },
  {
    runLabel: "Copilot Run 7", turnRange: "Turn 23-24",
    probingQuestions: [
      { text: "Ask if CM received suspicious emails or texts recently", status: "answered", target: "SCAM" },
      { text: "Check if there are additional unauthorized charges", status: "answered", target: "THIRD_PARTY_FRAUD" },
      { text: "Confirm whether card has chip or contactless enabled", status: "skipped", target: "THIRD_PARTY_FRAUD" },
      { text: "Probe for social engineering indicators", status: "answered", target: "SCAM" },
      { text: "Verify the $52.99 streaming charge details", status: "invalidated", target: "THIRD_PARTY_FRAUD" },
      { text: "Collect phishing text details: sender number, URL domain", status: "answered", target: "SCAM" },
      { text: "Ask when CM entered credentials on the site", status: "answered", target: "SCAM" },
      { text: "Check for additional compromised accounts", status: "answered", target: "SCAM" },
      { text: "Ask about password reuse across services", status: "answered", target: "SCAM" },
      { text: "Verify CM's location on date of in-store transaction", status: "answered", target: "THIRD_PARTY_FRAUD" },
      { text: "Document geographic impossibility as evidence", status: "pending", target: "THIRD_PARTY_FRAUD" },
    ],
    informationSufficient: false,
    caseEligibility: { fraudCase: "eligible", disputeCase: "blocked" },
  },
  {
    runLabel: "Copilot Run 8", turnRange: "Turn 25-27",
    probingQuestions: [
      { text: "Ask if CM received suspicious emails or texts recently", status: "answered", target: "SCAM" },
      { text: "Check if there are additional unauthorized charges", status: "answered", target: "THIRD_PARTY_FRAUD" },
      { text: "Confirm whether card has chip or contactless enabled", status: "skipped", target: "THIRD_PARTY_FRAUD" },
      { text: "Probe for social engineering indicators", status: "answered", target: "SCAM" },
      { text: "Verify the $52.99 streaming charge details", status: "invalidated", target: "THIRD_PARTY_FRAUD" },
      { text: "Collect phishing text details: sender number, URL domain", status: "answered", target: "SCAM" },
      { text: "Ask when CM entered credentials on the site", status: "answered", target: "SCAM" },
      { text: "Check for additional compromised accounts", status: "answered", target: "SCAM" },
      { text: "Ask about password reuse across services", status: "answered", target: "SCAM" },
      { text: "Verify CM's location on date of in-store transaction", status: "answered", target: "THIRD_PARTY_FRAUD" },
      { text: "Document geographic impossibility as evidence", status: "answered", target: "THIRD_PARTY_FRAUD" },
    ],
    informationSufficient: true,
    caseEligibility: { fraudCase: "eligible", disputeCase: "blocked" },
  },
  {
    runLabel: "Copilot Run 9", turnRange: "Turn 28-30",
    probingQuestions: [
      { text: "Ask if CM received suspicious emails or texts recently", status: "answered", target: "SCAM" },
      { text: "Check if there are additional unauthorized charges", status: "answered", target: "THIRD_PARTY_FRAUD" },
      { text: "Confirm whether card has chip or contactless enabled", status: "skipped", target: "THIRD_PARTY_FRAUD" },
      { text: "Probe for social engineering indicators", status: "answered", target: "SCAM" },
      { text: "Verify the $52.99 streaming charge details", status: "invalidated", target: "THIRD_PARTY_FRAUD" },
      { text: "Collect phishing text details: sender number, URL domain", status: "answered", target: "SCAM" },
      { text: "Ask when CM entered credentials on the site", status: "answered", target: "SCAM" },
      { text: "Check for additional compromised accounts", status: "answered", target: "SCAM" },
      { text: "Ask about password reuse across services", status: "answered", target: "SCAM" },
      { text: "Verify CM's location on date of in-store transaction", status: "answered", target: "THIRD_PARTY_FRAUD" },
      { text: "Document geographic impossibility as evidence", status: "answered", target: "THIRD_PARTY_FRAUD" },
    ],
    informationSufficient: true,
    caseEligibility: { fraudCase: "eligible", disputeCase: "blocked" },
  },
  {
    runLabel: "Copilot Run 10", turnRange: "Turn 31-35",
    probingQuestions: [
      { text: "Ask if CM received suspicious emails or texts recently", status: "answered", target: "SCAM" },
      { text: "Check if there are additional unauthorized charges", status: "answered", target: "THIRD_PARTY_FRAUD" },
      { text: "Confirm whether card has chip or contactless enabled", status: "skipped", target: "THIRD_PARTY_FRAUD" },
      { text: "Probe for social engineering indicators", status: "answered", target: "SCAM" },
      { text: "Verify the $52.99 streaming charge details", status: "invalidated", target: "THIRD_PARTY_FRAUD" },
      { text: "Collect phishing text details: sender number, URL domain", status: "answered", target: "SCAM" },
      { text: "Ask when CM entered credentials on the site", status: "answered", target: "SCAM" },
      { text: "Check for additional compromised accounts", status: "answered", target: "SCAM" },
      { text: "Ask about password reuse across services", status: "answered", target: "SCAM" },
      { text: "Verify CM's location on date of in-store transaction", status: "answered", target: "THIRD_PARTY_FRAUD" },
      { text: "Document geographic impossibility as evidence", status: "answered", target: "THIRD_PARTY_FRAUD" },
    ],
    informationSufficient: true,
    caseEligibility: { fraudCase: "eligible", disputeCase: "blocked" },
  },
];

// --- Score history for the curve chart ---

type ScorePoint = { label: string; scores: Record<string, number> };

const scoreHistory: ScorePoint[] = [
  { label: "Prior", scores: { "Third-Party Fraud": 0.25, "First-Party Fraud": 0.25, "Scam": 0.25, "Billing Dispute": 0.25 } },
  { label: "R1", scores: { "Third-Party Fraud": 0.72, "First-Party Fraud": 0.15, "Scam": 0.08, "Billing Dispute": 0.05 } },
  { label: "R2", scores: { "Third-Party Fraud": 0.68, "First-Party Fraud": 0.12, "Scam": 0.15, "Billing Dispute": 0.05 } },
  { label: "R3", scores: { "Third-Party Fraud": 0.45, "First-Party Fraud": 0.05, "Scam": 0.48, "Billing Dispute": 0.02 } },
  { label: "R4", scores: { "Third-Party Fraud": 0.38, "First-Party Fraud": 0.03, "Scam": 0.57, "Billing Dispute": 0.02 } },
  { label: "R5", scores: { "Third-Party Fraud": 0.32, "First-Party Fraud": 0.02, "Scam": 0.64, "Billing Dispute": 0.02 } },
  { label: "R6", scores: { "Third-Party Fraud": 0.28, "First-Party Fraud": 0.02, "Scam": 0.68, "Billing Dispute": 0.02 } },
  { label: "R7", scores: { "Third-Party Fraud": 0.22, "First-Party Fraud": 0.01, "Scam": 0.76, "Billing Dispute": 0.01 } },
  { label: "R8", scores: { "Third-Party Fraud": 0.18, "First-Party Fraud": 0.01, "Scam": 0.80, "Billing Dispute": 0.01 } },
  { label: "R9", scores: { "Third-Party Fraud": 0.15, "First-Party Fraud": 0.01, "Scam": 0.83, "Billing Dispute": 0.01 } },
  { label: "R10", scores: { "Third-Party Fraud": 0.12, "First-Party Fraud": 0.01, "Scam": 0.86, "Billing Dispute": 0.01 } },
];

const categoryColors: Record<string, string> = {
  "Third-Party Fraud": "#006FCF",
  "First-Party Fraud": "#F5A623",
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
    }}>{status}</span>
  );
}

function QuestionListPanel({ result }: { result: CopilotResult | null }) {
  if (!result) {
    return (
      <div style={{ padding: 20, color: "#53565A", fontSize: 14, textAlign: "center" }}>
        {"\u23F3"} Waiting for copilot...
      </div>
    );
  }

  const pending = result.probingQuestions.filter((pq) => pq.status === "pending");
  const resolved = result.probingQuestions.filter((pq) => pq.status !== "pending");

  return (
    <div style={{ padding: 16, display: "flex", flexDirection: "column", gap: 16 }}>
      {/* Pending questions — what CCP should ask */}
      <div>
        <div className="copilot-section-title" style={{ marginBottom: 8 }}>
          Pending ({pending.length})
        </div>
        {pending.length > 0 ? (
          <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
            {pending.map((pq, i) => (
              <li key={i} style={{ display: "flex", alignItems: "flex-start", gap: 8, marginBottom: 8, padding: "8px 10px", background: "#FFF3CD20", border: "1px solid #F5A62330", borderRadius: 8 }}>
                <QuestionStatusBadge status={pq.status} />
                <span style={{ fontSize: 15, lineHeight: 1.4 }}>
                  {pq.text}
                  <span style={{ color: "#53565A", fontSize: 13, marginLeft: 4 }}>[{pq.target}]</span>
                </span>
              </li>
            ))}
          </ul>
        ) : (
          <div style={{ fontSize: 15, color: "#53565A", fontStyle: "italic" }}>No pending questions</div>
        )}
      </div>

      {/* Resolved questions */}
      {resolved.length > 0 && (
        <div>
          <div className="copilot-section-title" style={{ marginBottom: 8 }}>
            Resolved ({resolved.length})
          </div>
          <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
            {resolved.map((pq, i) => (
              <li key={i} style={{ display: "flex", alignItems: "flex-start", gap: 8, marginBottom: 6, opacity: 0.8 }}>
                <QuestionStatusBadge status={pq.status} />
                <span style={{ fontSize: 14, lineHeight: 1.4, color: "#53565A" }}>
                  {pq.text}
                  <span style={{ fontSize: 12, marginLeft: 4 }}>[{pq.target}]</span>
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

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
            <QuestionListPanel result={activeRun >= 0 ? copilotResults[activeRun] : null} />
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
                <CaseEligibilityPanel result={copilotResults[activeRun]} isActive={true} />
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
