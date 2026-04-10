import React from "react";

// --- Layer data ---

type LayerItem = {
  label: string;
  icon: string;
};

type HorizontalLayer = {
  name: string;
  subtitle: string;
  color: string;
  items: LayerItem[];
};

const layers: HorizontalLayer[] = [
  {
    name: "Interaction Layer",
    subtitle: "User-facing surface",
    color: "#006FCF",
    items: [
      { label: "CCP Desktop", icon: "\uD83C\uDFA7" },
      { label: "Conversation Stream", icon: "\uD83C\uDFA4" },
      { label: "Copilot Suggestions", icon: "\u2728" },
      { label: "Case Summary Report", icon: "\uD83D\uDCCB" },
    ],
  },
  {
    name: "Copilot Layer",
    subtitle: "Multi-agent workflow",
    color: "#00175A",
    items: [
      { label: "Orchestrator", icon: "\uD83C\uDFAF" },
      { label: "Retrieval Agent", icon: "\uD83D\uDD0D" },
      { label: "Specialist Agents", icon: "\uD83D\uDEE1\uFE0F" },
      { label: "Arbitrator", icon: "\u2696\uFE0F" },
      { label: "Case Advisor", icon: "\uD83D\uDCA1" },
    ],
  },
  {
    name: "Knowledge Layer",
    subtitle: "Policies & business rules",
    color: "#0050A0",
    items: [
      { label: "Investigation Taxonomy", icon: "\uD83D\uDCD6" },
      { label: "Case Opening Rules", icon: "\uD83D\uDCDC" },
      { label: "Compliance Policies", icon: "\uD83D\uDD10" },
      { label: "Scheme Reason Codes", icon: "\uD83C\uDFF7\uFE0F" },
    ],
  },
  {
    name: "Data Layer",
    subtitle: "Real-time fetching from reliable data sources",
    color: "#008000",
    items: [
      { label: "Tool Gateway", icon: "\uD83D\uDD27" },
      { label: "Transaction Data", icon: "\uD83D\uDCB3" },
      { label: "Customer Profiles", icon: "\uD83D\uDC64" },
      { label: "Auth Logs", icon: "\uD83D\uDCDD" },
      { label: "Evidence Store", icon: "\uD83D\uDDC4\uFE0F" },
    ],
  },
];

type SideBar = {
  name: string;
  subtitle: string;
  color: string;
  items: LayerItem[];
};

const observability: SideBar = {
  name: "Observability",
  subtitle: "Production monitoring",
  color: "#6B21A8",
  items: [
    { label: "LangFuse Tracing", icon: "\uD83D\uDD2D" },
    { label: "Latency Monitoring", icon: "\u23F1\uFE0F" },
    { label: "Token Usage", icon: "\uD83D\uDCCA" },
    { label: "Error Tracking", icon: "\u26A0\uFE0F" },
    { label: "Audit Logs", icon: "\uD83D\uDCDD" },
    { label: "Cost Analytics", icon: "\uD83D\uDCB0" },
  ],
};

const evaluation: SideBar = {
  name: "Evaluation",
  subtitle: "Development & tuning",
  color: "#CF291D",
  items: [
    { label: "Copilot Scoring", icon: "\uD83C\uDFAF" },
    { label: "Accuracy Metrics", icon: "\uD83D\uDCCF" },
    { label: "Hypothesis Quality", icon: "\uD83E\uDDEA" },
    { label: "Regression Tests", icon: "\uD83E\uDDEE" },
    { label: "Prompt Assessment", icon: "\uD83D\uDCA1" },
    { label: "A/B Testing", icon: "\u2696\uFE0F" },
  ],
};

// --- Components ---

function LayerCard({ layer }: { layer: HorizontalLayer }) {
  return (
    <div
      style={{
        background: "#ffffff",
        border: `2px solid ${layer.color}`,
        borderRadius: 14,
        padding: "14px 24px",
        display: "flex",
        flexDirection: "column",
        gap: 10,
      }}
    >
      <div style={{ display: "flex", alignItems: "baseline", gap: 12 }}>
        <div style={{ fontSize: 18, fontWeight: 700, color: layer.color, whiteSpace: "nowrap" }}>
          {layer.name}
        </div>
        <div style={{ fontSize: 14, color: "#53565A" }}>
          {layer.subtitle}
        </div>
      </div>
      <div style={{ display: "flex", flexWrap: "nowrap", gap: 10 }}>
        {layer.items.map((item) => (
          <div
            key={item.label}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 6,
              background: `${layer.color}0A`,
              border: `1px solid ${layer.color}25`,
              borderRadius: 8,
              padding: "6px 14px",
              fontSize: 15,
              fontWeight: 500,
              color: "#00175A",
              whiteSpace: "nowrap",
            }}
          >
            <span style={{ fontSize: 16 }}>{item.icon}</span>
            {item.label}
          </div>
        ))}
      </div>
    </div>
  );
}

function SideBarCard({ bar }: { bar: SideBar }) {
  return (
    <div
      style={{
        background: "#ffffff",
        border: `2px solid ${bar.color}`,
        borderRadius: 14,
        padding: "16px 16px",
        display: "flex",
        flexDirection: "column",
        alignSelf: "stretch",
        width: 200,
      }}
    >
      <div style={{ textAlign: "center", marginBottom: 10 }}>
        <div style={{ fontSize: 17, fontWeight: 700, color: bar.color }}>
          {bar.name}
        </div>
        <div style={{ fontSize: 12, color: "#53565A", marginTop: 3 }}>
          {bar.subtitle}
        </div>
      </div>
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          flex: 1,
          justifyContent: "space-evenly",
        }}
      >
        {bar.items.map((item) => (
          <div
            key={item.label}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 6,
              background: `${bar.color}0A`,
              border: `1px solid ${bar.color}25`,
              borderRadius: 8,
              padding: "7px 12px",
              fontSize: 14,
              fontWeight: 500,
              color: "#00175A",
            }}
          >
            <span style={{ fontSize: 15 }}>{item.icon}</span>
            {item.label}
          </div>
        ))}
      </div>
    </div>
  );
}

// --- Main Page ---

export default function LayersPage() {
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
          System Architecture
          <span style={{ fontSize: 14, color: "#53565A", fontWeight: 400, marginLeft: 12 }}>
            Layered Architecture View
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
            href="#/demo"
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
            Live Demo
          </a>
        </div>
      </div>

      {/* Diagram: sidebar | center stack | sidebar */}
      <div
        style={{
          flex: 1,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          padding: "20px 40px",
          overflow: "auto",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "stretch",
            gap: 18,
          }}
        >
          {/* Left sidebar — Observability */}
          <SideBarCard bar={observability} />

          {/* Center — horizontal layer stack */}
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              justifyContent: "center",
              gap: 0,
            }}
          >
            {layers.map((layer, i) => (
              <React.Fragment key={layer.name}>
                <LayerCard layer={layer} />
                {i < layers.length - 1 && (
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "center",
                      padding: "4px 0",
                      color: "#97999B",
                      fontSize: 20,
                    }}
                  >
                    {"\u2195"}
                  </div>
                )}
              </React.Fragment>
            ))}
          </div>

          {/* Right sidebar — Evaluation */}
          <SideBarCard bar={evaluation} />
        </div>
      </div>
    </div>
  );
}
