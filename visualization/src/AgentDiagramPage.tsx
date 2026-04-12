import React, { useState, useEffect } from "react";

// --- Agent diagram data ---

type IOItem = {
  label: string;
  icon: string;
  realtime?: boolean;
};

type AgentDiagram = {
  id: string;
  name: string;
  icon: string;
  color: string;
  subtitle: string;
  description: string;
  inputs: IOItem[];
  outputs: IOItem[];
};

const agents: AgentDiagram[] = [
  {
    id: "triage",
    name: "Intent Extractor",
    icon: "\uD83D\uDCCB",
    color: "#006FCF",
    subtitle: "",
    description: "",
    inputs: [
      { label: "Conversation Transcript", icon: "\uD83D\uDCAC" },
      { label: "Previous Allegations", icon: "\uD83D\uDCC4" },
    ],
    outputs: [
      { label: "Structured Allegations", icon: "\uD83D\uDCCB" },
      { label: "Allegation Type & Confidence", icon: "\uD83C\uDFAF" },
      { label: "Key Entities (merchant, amount, date)", icon: "\uD83D\uDD0D" },
    ],
  },
  {
    id: "auth",
    name: "Auth Agent",
    icon: "\uD83D\uDD12",
    color: "#006FCF",
    subtitle: "",
    description: "",
    inputs: [
      { label: "Conversation Transcript", icon: "\uD83D\uDCAC" },
      { label: "Authentication Logs", icon: "\uD83D\uDCDD" },
      { label: "Customer Profile", icon: "\uD83D\uDC64" },
    ],
    outputs: [
      { label: "Impersonation Risk Score", icon: "\u26A0\uFE0F" },
      { label: "Risk Factors", icon: "\uD83D\uDEA9" },
      { label: "Step-up Recommendation", icon: "\uD83D\uDD10" },
    ],
  },
  {
    id: "retrieval",
    name: "Retrieval Agent",
    icon: "\uD83D\uDD0D",
    color: "#006FCF",
    subtitle: "",
    description: "",
    inputs: [
      { label: "Case Identifier", icon: "\uD83D\uDCC1" },
    ],
    outputs: [
      { label: "Transaction Records", icon: "\uD83D\uDCB3", realtime: true },
      { label: "Authentication Events", icon: "\uD83D\uDD12", realtime: true },
      { label: "Customer Profile", icon: "\uD83D\uDC64", realtime: true },
      { label: "Data Gaps Identified", icon: "\u2753" },
    ],
  },
  {
    id: "specialists",
    name: "Specialist Panel",
    icon: "\uD83D\uDEE1\uFE0F",
    color: "#006FCF",
    subtitle: "",
    description: "",
    inputs: [
      { label: "Accumulated Allegations", icon: "\uD83D\uDCCB" },
      { label: "Retrieved Evidence", icon: "\uD83D\uDD0D" },
      { label: "Conversation Context", icon: "\uD83D\uDCAC" },
      { label: "Investigation Policies", icon: "\uD83D\uDCDC" },
    ],
    outputs: [
      { label: "Likelihood per Category", icon: "\uD83D\uDCCA" },
      { label: "Case Eligibility (eligible/blocked)", icon: "\u2705" },
      { label: "Evidence Gaps", icon: "\u2753" },
      { label: "Policy Citations", icon: "\uD83D\uDCDC" },
    ],
  },
  {
    id: "arbitrator",
    name: "Typing Arbitrator",
    icon: "\u2696\uFE0F",
    color: "#006FCF",
    subtitle: "",
    description: "",
    inputs: [
      { label: "Specialist Assessments", icon: "\uD83D\uDEE1\uFE0F" },
      { label: "Prior Case Typing Scores", icon: "\uD83D\uDCCA" },
      { label: "Auth Assessment", icon: "\uD83D\uDD12" },
    ],
    outputs: [
      { label: "Updated Case Typing Scores", icon: "\uD83C\uDFAF" },
      { label: "Per-category Reasoning", icon: "\uD83D\uDCDD" },
      { label: "Detected Contradictions", icon: "\u26A0\uFE0F" },
    ],
  },
  {
    id: "advisor",
    name: "Case Advisor",
    icon: "\uD83D\uDCA1",
    color: "#006FCF",
    subtitle: "",
    description: "",
    inputs: [
      { label: "Specialist Evidence Gaps", icon: "\u2753" },
      { label: "Case Typing Scores", icon: "\uD83D\uDCCA" },
      { label: "Validated Question List (with statuses)", icon: "\uD83D\uDCCB" },
    ],
    outputs: [
      { label: "New Probing Questions (0-3)", icon: "\uD83D\uDDE3\uFE0F" },
      { label: "Question Targets (per question)", icon: "\uD83C\uDFAF" },
      { label: "Case Eligibility Status", icon: "\u2705" },
    ],
  },
  {
    id: "question_validator",
    name: "Question Validator",
    icon: "\u2753",
    color: "#006FCF",
    subtitle: "",
    description: "",
    inputs: [
      { label: "Pending Probing Questions", icon: "\uD83D\uDCCB" },
      { label: "New Conversation Turns", icon: "\uD83D\uDCAC" },
      { label: "Current Hypothesis Scores", icon: "\uD83D\uDCCA" },
    ],
    outputs: [
      { label: "Status Updates (answered/invalidated)", icon: "\u2705" },
      { label: "Resolution Reasons", icon: "\uD83D\uDCDD" },
    ],
  },
];

// --- Components ---

function Arrow({ direction, color }: { direction: "right" | "left"; color: string }) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        width: 60,
        flexShrink: 0,
      }}
    >
      <div
        style={{
          fontSize: 28,
          color,
          lineHeight: 1,
        }}
      >
        {direction === "right" ? "\u2192" : "\u2190"}
      </div>
    </div>
  );
}

function IOColumn({ items, side, color }: { items: IOItem[]; side: "input" | "output"; color: string }) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 12,
        flex: 1,
        maxWidth: 300,
      }}
    >
      <div
        style={{
          fontSize: 13,
          fontWeight: 700,
          color: "#53565A",
          textTransform: "uppercase",
          letterSpacing: 1.2,
          textAlign: "center",
          marginBottom: 4,
        }}
      >
        {side === "input" ? "Inputs" : "Outputs"}
      </div>
      {items.map((item) => (
        <div
          key={item.label}
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            background: `${color}08`,
            border: `1.5px solid ${color}30`,
            borderRadius: 10,
            padding: "10px 16px",
            fontSize: 15,
            fontWeight: 500,
            color: "#00175A",
          }}
        >
          <span style={{ fontSize: 18, flexShrink: 0 }}>{item.icon}</span>
          {item.label}
        </div>
      ))}
    </div>
  );
}

function AgentCenter({ agent }: { agent: AgentDiagram }) {
  const isSpecialistPanel = agent.id === "specialists";

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 12,
        width: 300,
        flexShrink: 0,
      }}
    >
      <div
        style={{
          background: "#ffffff",
          border: `3px solid ${agent.color}`,
          borderRadius: 16,
          padding: "20px 24px",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 10,
          width: "100%",
          boxShadow: `0 4px 20px ${agent.color}15`,
        }}
      >
        <span style={{ fontSize: 36 }}>{agent.icon}</span>
        <div style={{ fontSize: 20, fontWeight: 700, color: agent.color, textAlign: "center" }}>
          {agent.name}
        </div>
        {agent.subtitle && (
          <div
            style={{
              fontSize: 12,
              fontWeight: 600,
              color: "#ffffff",
              background: agent.color,
              padding: "3px 12px",
              borderRadius: 6,
              letterSpacing: 0.5,
            }}
          >
            {agent.subtitle}
          </div>
        )}
        {isSpecialistPanel && (
          <div style={{ display: "flex", gap: 8, marginTop: 4, width: "100%" }}>
            {[
              { label: "Dispute", icon: "\uD83D\uDCE6" },
              { label: "Scam", icon: "\uD83C\uDFA3" },
              { label: "Fraud", icon: "\uD83D\uDEE1\uFE0F" },
            ].map((s) => (
              <div
                key={s.label}
                style={{
                  flex: 1,
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  gap: 4,
                  background: `${agent.color}08`,
                  border: `1.5px solid ${agent.color}25`,
                  borderRadius: 10,
                  padding: "10px 6px",
                }}
              >
                <span style={{ fontSize: 22 }}>{s.icon}</span>
                <span style={{ fontSize: 12, fontWeight: 600, color: "#00175A" }}>{s.label}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function FlowBox({ icon, label, sublabel, color, size }: { icon: string; label: string; sublabel?: string; color: string; size?: "lg" | "md" | "sm" }) {
  const isLg = size === "lg";
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: isLg ? 6 : 4,
        background: "#ffffff",
        border: `${isLg ? 3 : 2}px solid ${color}`,
        borderRadius: isLg ? 16 : 12,
        padding: isLg ? "16px 24px" : "10px 18px",
        boxShadow: isLg ? `0 4px 20px ${color}15` : undefined,
        minWidth: isLg ? 160 : undefined,
      }}
    >
      <span style={{ fontSize: isLg ? 32 : 24 }}>{icon}</span>
      <div style={{ fontSize: isLg ? 18 : 14, fontWeight: 700, color, textAlign: "center" }}>{label}</div>
      {sublabel && <div style={{ fontSize: 11, color: "#53565A", textAlign: "center" }}>{sublabel}</div>}
    </div>
  );
}

function FlowArrow({ color }: { color: string }) {
  return (
    <div style={{ display: "flex", alignItems: "center", flexShrink: 0, padding: "0 4px" }}>
      <div style={{ width: 30, height: 2, background: color }} />
      <div style={{ fontSize: 22, color, lineHeight: 1, marginLeft: -4 }}>{"\u25B6"}</div>
    </div>
  );
}

function RetrievalDiagram({ agent }: { agent: AgentDiagram }) {
  const c = agent.color;
  const results = [
    { label: "Transaction Records", icon: "\uD83D\uDCB3" },
    { label: "Auth Events", icon: "\uD83D\uDD12" },
    { label: "Customer Profile", icon: "\uD83D\uDC64" },
    { label: "Data Gaps", icon: "\u2753" },
  ];

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 24,
        maxWidth: 1100,
        width: "100%",
      }}
    >
      {/* Main horizontal flow */}
      <div style={{ display: "flex", alignItems: "center", gap: 0 }}>
        {/* Case Identifier (input style) */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 12,
            maxWidth: 300,
          }}
        >
          <div
            style={{
              fontSize: 13,
              fontWeight: 700,
              color: "#53565A",
              textTransform: "uppercase",
              letterSpacing: 1.2,
              textAlign: "center",
              marginBottom: 4,
            }}
          >
            Inputs
          </div>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              background: `${c}08`,
              border: `1.5px solid ${c}30`,
              borderRadius: 10,
              padding: "10px 16px",
              fontSize: 15,
              fontWeight: 500,
              color: "#00175A",
            }}
          >
            <span style={{ fontSize: 18, flexShrink: 0 }}>{"\uD83D\uDCC1"}</span>
            Case Identifier
          </div>
        </div>
        <Arrow direction="right" color={c} />

        {/* Retrieval Agent */}
        <FlowBox icon={agent.icon} label={agent.name} color={c} size="lg" />
        <FlowArrow color={c} />

        {/* Tool Gateway */}
        <FlowBox icon={"\uD83D\uDD27"} label="Tool Gateway" sublabel="Selects data tools" color={c} />
        <FlowArrow color={c} />

        {/* Database */}
        <FlowBox icon={"\uD83D\uDDC4\uFE0F"} label="Database" sublabel="Real-time fetch" color="#00175A" />
        <FlowArrow color={c} />

        {/* Results */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 8,
          }}
        >
          <div
            style={{
              fontSize: 12,
              fontWeight: 700,
              color: "#53565A",
              textTransform: "uppercase",
              letterSpacing: 1,
              textAlign: "center",
              marginBottom: 2,
            }}
          >
            Returns
          </div>
          {results.map((r) => (
            <div
              key={r.label}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 6,
                background: `${c}08`,
                border: `1.5px solid ${c}30`,
                borderRadius: 8,
                padding: "6px 12px",
                fontSize: 13,
                fontWeight: 500,
                color: "#00175A",
              }}
            >
              <span style={{ fontSize: 15 }}>{r.icon}</span>
              {r.label}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function AgentDiagramCard({ agent }: { agent: AgentDiagram }) {
  if (agent.id === "retrieval") {
    return <RetrievalDiagram agent={agent} />;
  }

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        gap: 0,
        width: "100%",
        maxWidth: 1100,
      }}
    >
      <IOColumn items={agent.inputs} side="input" color={agent.color} />
      <Arrow direction="right" color={agent.color} />
      <AgentCenter agent={agent} />
      <Arrow direction="right" color={agent.color} />
      <IOColumn items={agent.outputs} side="output" color={agent.color} />
    </div>
  );
}

// --- Main Page ---

export default function AgentDiagramPage() {
  const [currentIdx, setCurrentIdx] = useState(0);

  // Support direct hash navigation like #/agent/triage
  useEffect(() => {
    const hash = window.location.hash;
    const match = hash.match(/#\/agent\/(\w+)/);
    if (match) {
      const idx = agents.findIndex((a) => a.id === match[1]);
      if (idx >= 0) setCurrentIdx(idx);
    }
  }, []);

  const agent = agents[currentIdx];

  return (
    <div
      style={{
        width: "100vw",
        height: "100vh",
        display: "flex",
        flexDirection: "column",
        background: "#ffffff",
        color: "#00175A",
        overflow: "hidden",
      }}
    >
      {/* Header */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "12px 24px",
          borderBottom: "1px solid #E0E0E0",
          flexShrink: 0,
        }}
      >
        <div style={{ fontSize: 22, fontWeight: 700, color: "#00175A" }}>
          Agent Detail
          <span style={{ fontSize: 14, color: "#53565A", fontWeight: 400, marginLeft: 12 }}>
            {agent.name}
          </span>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <a
            href="#/"
            style={{
              background: "#F7F8F9",
              border: "1px solid #E0E0E0",
              borderRadius: 6,
              color: "#00175A",
              fontSize: 14,
              fontWeight: 600,
              padding: "8px 16px",
              textDecoration: "none",
            }}
          >
            Workflow
          </a>
          <a
            href="#/layers"
            style={{
              background: "#F7F8F9",
              border: "1px solid #E0E0E0",
              borderRadius: 6,
              color: "#00175A",
              fontSize: 14,
              fontWeight: 600,
              padding: "8px 16px",
              textDecoration: "none",
            }}
          >
            Architecture
          </a>
        </div>
      </div>

      {/* Agent selector tabs */}
      <div
        style={{
          display: "flex",
          gap: 6,
          padding: "12px 24px",
          borderBottom: "1px solid #E0E0E0",
          flexShrink: 0,
          overflowX: "auto",
        }}
      >
        {agents.map((a, i) => (
          <button
            key={a.id}
            onClick={() => setCurrentIdx(i)}
            style={{
              background: i === currentIdx ? a.color : "#F7F8F9",
              border: `1.5px solid ${i === currentIdx ? a.color : "#E0E0E0"}`,
              borderRadius: 8,
              color: i === currentIdx ? "#ffffff" : "#00175A",
              fontSize: 14,
              fontWeight: 600,
              padding: "8px 16px",
              cursor: "pointer",
              whiteSpace: "nowrap",
              transition: "all 0.2s ease",
            }}
          >
            <span style={{ marginRight: 6 }}>{a.icon}</span>
            {a.name}
          </button>
        ))}
      </div>

      {/* Diagram */}
      <div
        style={{
          flex: 1,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          padding: "24px 40px",
        }}
      >
        <AgentDiagramCard agent={agent} />
      </div>
    </div>
  );
}
