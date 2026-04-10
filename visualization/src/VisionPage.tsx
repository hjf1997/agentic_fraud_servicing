import React from "react";
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  BackgroundVariant,
  type Node,
  type Edge,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import AgentNode from "./components/AgentNode";
import GroupLabel from "./components/GroupLabel";

const nodeTypes = { agent: AgentNode, group: GroupLabel };

const C = {
  shared: "#006FCF",
  dispute: "#008000",
  fraud: "#00175A",
  scam: "#CF291D",
  future: "#97999B",
  group: "#53565A",
};

const nodes: Node[] = [
  // Group: Shared Foundation
  {
    id: "group_shared",
    type: "group",
    position: { x: 0, y: 0 },
    data: {
      label: "Shared Foundation",
      color: C.shared,
      width: 280,
      height: 230,
    },
    draggable: false,
    selectable: false,
  },

  // Shared components
  {
    id: "orchestration",
    type: "agent",
    position: { x: 40, y: 25 },
    data: {
      label: "Orchestration",
      icon: "\uD83C\uDFAF",
      color: C.shared,
      status: "done",
      compact: true,
    },
  },
  {
    id: "retrieval",
    type: "agent",
    position: { x: 40, y: 75 },
    data: {
      label: "Evidence Retrieval",
      icon: "\uD83D\uDD0D",
      color: C.shared,
      status: "done",
      compact: true,
    },
  },
  {
    id: "scoring",
    type: "agent",
    position: { x: 40, y: 125 },
    data: {
      label: "Case Type Scoring",
      icon: "\uD83D\uDCCA",
      color: C.shared,
      status: "done",
      compact: true,
    },
  },
  {
    id: "observability",
    type: "agent",
    position: { x: 40, y: 175 },
    data: {
      label: "Observability",
      icon: "\uD83D\uDD2D",
      color: C.shared,
      status: "done",
      compact: true,
    },
  },

  // Group: Category-Specific Knowledge
  {
    id: "group_knowledge",
    type: "group",
    position: { x: 420, y: 0 },
    data: {
      label: "Category-Specific Knowledge",
      color: C.group,
      width: 280,
      height: 230,
    },
    draggable: false,
    selectable: false,
  },

  // Category modules
  {
    id: "dispute_knowledge",
    type: "agent",
    position: { x: 460, y: 25 },
    data: {
      label: "Billing Dispute",
      icon: "\uD83D\uDCE6",
      color: C.dispute,
      status: "done",
      compact: true,
    },
  },
  {
    id: "fraud_knowledge",
    type: "agent",
    position: { x: 460, y: 75 },
    data: {
      label: "Fraud",
      icon: "\uD83D\uDEE1\uFE0F",
      color: C.fraud,
      status: "done",
      compact: true,
    },
  },
  {
    id: "scam_knowledge",
    type: "agent",
    position: { x: 460, y: 125 },
    data: {
      label: "Scam",
      icon: "\uD83C\uDFA3",
      color: C.scam,
      status: "done",
      compact: true,
    },
  },
  {
    id: "future_knowledge",
    type: "agent",
    position: { x: 460, y: 175 },
    data: {
      label: "New Category",
      icon: "\u2795",
      color: C.future,
      status: "idle",
      compact: true,
    },
  },

  // CCP Desktop output
  {
    id: "ccp",
    type: "agent",
    position: { x: 250, y: 242 },
    data: {
      label: "CCP Desktop",
      icon: "\uD83C\uDFA7",
      color: C.shared,
      status: "done",
      compact: true,
    },
  },
];

const makeEdge = (
  source: string,
  target: string,
  color: string,
  dashed?: boolean
): Edge => ({
  id: `${source}-${target}`,
  source,
  sourceHandle: "right",
  target,
  targetHandle: "left",
  animated: !dashed,
  style: {
    stroke: color,
    strokeWidth: 2,
    strokeDasharray: dashed ? "6 4" : undefined,
    opacity: dashed ? 0.4 : 0.7,
  },
});

const edges: Edge[] = [
  // Shared → each category
  makeEdge("orchestration", "dispute_knowledge", C.dispute),
  makeEdge("orchestration", "fraud_knowledge", C.fraud),
  makeEdge("orchestration", "scam_knowledge", C.scam),
  makeEdge("orchestration", "future_knowledge", C.future, true),

  makeEdge("retrieval", "dispute_knowledge", C.dispute),
  makeEdge("retrieval", "fraud_knowledge", C.fraud),
  makeEdge("retrieval", "scam_knowledge", C.scam),
  makeEdge("retrieval", "future_knowledge", C.future, true),

  makeEdge("scoring", "dispute_knowledge", C.dispute),
  makeEdge("scoring", "fraud_knowledge", C.fraud),
  makeEdge("scoring", "scam_knowledge", C.scam),
  makeEdge("scoring", "future_knowledge", C.future, true),

  // Shared → CCP
  {
    id: "scoring-ccp",
    source: "scoring",
    target: "ccp",
    animated: true,
    style: { stroke: C.shared, strokeWidth: 2, opacity: 0.7 },
  },

  // Categories → CCP
  {
    id: "dispute_knowledge-ccp",
    source: "dispute_knowledge",
    target: "ccp",
    animated: true,
    style: { stroke: C.dispute, strokeWidth: 2, opacity: 0.7 },
  },
  {
    id: "fraud_knowledge-ccp",
    source: "fraud_knowledge",
    target: "ccp",
    animated: true,
    style: { stroke: C.fraud, strokeWidth: 2, opacity: 0.7 },
  },
  {
    id: "scam_knowledge-ccp",
    source: "scam_knowledge",
    target: "ccp",
    animated: true,
    style: { stroke: C.scam, strokeWidth: 2, opacity: 0.7 },
  },
];

function VisionInner() {
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
          Copilot Vision
          <span
            style={{
              fontSize: 14,
              color: "#53565A",
              fontWeight: 400,
              marginLeft: 12,
            }}
          >
            One Unified Framework for All Case Types
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

      {/* Diagram */}
      <div style={{ flex: 1, position: "relative" }}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          fitView
          fitViewOptions={{ padding: 0.3 }}
          nodesDraggable={false}
          nodesConnectable={false}
          edgesReconnectable={false}
          deleteKeyCode={null}
          proOptions={{ hideAttribution: true }}
          minZoom={0.5}
          maxZoom={1.5}
          panOnDrag={false}
          zoomOnScroll={false}
          zoomOnPinch={false}
          zoomOnDoubleClick={false}
        >
          <Background
            variant={BackgroundVariant.Dots}
            color="#C8C9C7"
            gap={20}
            size={1}
          />
        </ReactFlow>
      </div>
    </div>
  );
}

export default function VisionPage() {
  return (
    <ReactFlowProvider>
      <VisionInner />
    </ReactFlowProvider>
  );
}
