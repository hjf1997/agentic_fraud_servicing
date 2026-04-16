import type { Node, Edge } from "@xyflow/react";
import type { AgentNodeData } from "./components/AgentNode";
import type { GroupLabelData } from "./components/GroupLabel";

// Colors by role
const C = {
  input: "#006FCF", // AMEX blue
  orchestrator: "#00175A", // AMEX dark blue
  stage1: "#006FCF", // AMEX blue
  specialist: "#008000", // green
  arbitrator: "#CF291D", // AMEX red
  advisor: "#006FCF", // AMEX blue
  output: "#00175A", // AMEX dark blue
  knowledge: "#B7791F", // Knowledge layer gold
  group: "#53565A", // AMEX grey
};

type AgentNode = Node<AgentNodeData, "agent">;
type GroupNode = Node<GroupLabelData, "group">;

// --- Nodes ---

// Conversation bar node — rendered inside React Flow so it pans/zooms with the graph
const conversationBarNode: Node = {
  id: "conversation_bar",
  type: "conversationBar",
  position: { x: 0, y: -70 },
  data: { stepIdx: 0 },
};

const agentNodes: AgentNode[] = [
  // Transcript stream + Orchestrator (same row)
  {
    id: "transcript",
    type: "agent",
    position: { x: 140, y: 30 },
    data: {
      label: "Conversation Intake",
      subtitle: "Turns received since last copilot",
      icon: "\uD83C\uDFA4",
      color: C.input,
    },
  },
  {
    id: "orchestrator",
    type: "agent",
    position: { x: 520, y: 30 },
    data: {
      label: "Copilot Orchestrator",
      subtitle: "Decides which agents to trigger",
      icon: "\uD83C\uDFAF",
      color: C.orchestrator,
    },
  },

  // Stage 1: parallel
  {
    id: "triage",
    type: "agent",
    position: { x: 170, y: 140 },
    data: {
      label: "Intent Extractor",
      subtitle: "17-type intent taxonomy",
      icon: "\uD83D\uDCCB",
      color: C.stage1,
    },
  },
  {
    id: "retrieval",
    type: "agent",
    position: { x: 430, y: 140 },
    data: {
      label: "Retrieval Agent",
      subtitle: "Structured data including transactions",
      icon: "\uD83D\uDD0D",
      color: C.stage1,
    },
  },

  // Stage 2: specialists (parallel)
  {
    id: "dispute_spec",
    type: "agent",
    position: { x: 70, y: 250 },
    data: {
      label: "Dispute Specialist",
      subtitle: "Merchant performance claims",
      icon: "\uD83D\uDCE6",
      color: C.specialist,
    },
  },
  {
    id: "scam_spec",
    type: "agent",
    position: { x: 310, y: 250 },
    data: {
      label: "Scam Specialist",
      subtitle: "External manipulator detection",
      icon: "\uD83C\uDFA3",
      color: C.specialist,
    },
  },
  {
    id: "fraud_spec",
    type: "agent",
    position: { x: 550, y: 250 },
    data: {
      label: "Fraud Specialist",
      subtitle: "Unauthorized access / compromise",
      icon: "\uD83D\uDEE1\uFE0F",
      color: C.specialist,
    },
  },

  // Knowledge Database — single node feeding into specialists
  {
    id: "knowledge_db",
    type: "agent",
    position: { x: 780, y: 155 },
    data: {
      label: "Knowledge Database",
      subtitle: "Historical case summary",
      icon: "\uD83D\uDCD6",
      color: C.knowledge,
    },
  },

  // Arbitrator + Advisor (parallel)
  {
    id: "arbitrator",
    type: "agent",
    position: { x: 170, y: 365 },
    data: {
      label: "Typing Arbitrator",
      subtitle: "4-category Bayesian scoring",
      icon: "\u2696\uFE0F",
      color: C.arbitrator,
    },
  },
  {
    id: "advisor",
    type: "agent",
    position: { x: 460, y: 365 },
    data: {
      label: "Case Advisor",
      subtitle: "Probing questions & case eligibility",
      icon: "\uD83D\uDCA1",
      color: C.arbitrator,
    },
  },

  // Output
  {
    id: "suggestion",
    type: "agent",
    position: { x: 310, y: 460 },
    data: {
      label: "Copilot Suggestion",
      subtitle: "Scores, questions, risk flags, eligibility",
      icon: "\u2728",
      color: C.output,
    },
  },

  // CCP desktop
  {
    id: "ccp_output",
    type: "agent",
    position: { x: 700, y: 460 },
    data: {
      label: "CCP Desktop",
      subtitle: "Suggestion delivered to agent",
      icon: "\uD83C\uDFA7",
      color: C.advisor,
    },
  },
];

const groupNodes: GroupNode[] = [
  {
    id: "group_phase1",
    type: "group",
    position: { x: 130, y: 120 },
    data: {
      label: "Stage 1 \u2014 Retrieval & Triage",
      color: C.stage1,
      width: 600,
      height: 95,
    },
    draggable: false,
    selectable: false,
  },
  {
    id: "group_phase2",
    type: "group",
    position: { x: 30, y: 230 },
    data: {
      label: "Stage 2 \u2014 Specialist Assessment",
      color: C.specialist,
      width: 780,
      height: 95,
    },
    draggable: false,
    selectable: false,
  },
  {
    id: "group_synth",
    type: "group",
    position: { x: 130, y: 345 },
    data: {
      label: "Stage 3 \u2014 Aggregation & Decision",
      color: C.arbitrator,
      width: 600,
      height: 95,
    },
    draggable: false,
    selectable: false,
  },
];

export const initialNodes: Node[] = [
  conversationBarNode,
  ...groupNodes,
  ...agentNodes,
];

// --- Edges ---

const makeEdge = (
  source: string,
  target: string,
  label?: string,
  color?: string,
): Edge => ({
  id: `${source}-${target}`,
  source,
  target,
  label,
  animated: false,
  style: { stroke: color || C.group, strokeWidth: 2, opacity: 0.3 },
  labelStyle: { fill: "#53565A", fontSize: 9, fontWeight: 600 },
  labelBgStyle: { fill: "#ffffff", fillOpacity: 0.9 },
  labelBgPadding: [6, 3] as [number, number],
  labelBgBorderRadius: 4,
});

const knowledgeEdges = ["dispute_spec", "scam_spec", "fraud_spec"].map(
  (spec) => `knowledge_db-${spec}`
);

export const initialEdges: Edge[] = [
  // Input -> Orchestrator (horizontal, right to left)
  {
    id: "transcript-orchestrator",
    source: "transcript",
    sourceHandle: "right",
    target: "orchestrator",
    targetHandle: "left",
    animated: false,
    style: { stroke: C.input, strokeWidth: 2, opacity: 0.3 },
    labelStyle: { fill: "#53565A", fontSize: 9, fontWeight: 600 },
    labelBgStyle: { fill: "#ffffff", fillOpacity: 0.9 },
    labelBgPadding: [6, 3] as [number, number],
    labelBgBorderRadius: 4,
  },

  // Orchestrator -> Stage 1
  makeEdge("orchestrator", "triage", undefined, C.stage1),
  makeEdge("orchestrator", "retrieval", undefined, C.stage1),

  // Stage 1 -> Stage 2 (Specialists)
  makeEdge("triage", "dispute_spec", undefined, C.specialist),
  makeEdge("triage", "scam_spec", undefined, C.specialist),
  makeEdge("triage", "fraud_spec", undefined, C.specialist),
  makeEdge("retrieval", "dispute_spec", undefined, C.specialist),
  makeEdge("retrieval", "scam_spec", undefined, C.specialist),
  makeEdge("retrieval", "fraud_spec", undefined, C.specialist),


  // Knowledge Database -> Specialists (dashed)
  ...(["dispute_spec", "scam_spec", "fraud_spec"] as const).map(
    (spec): Edge => ({
      id: `knowledge_db-${spec}`,
      source: "knowledge_db",
      sourceHandle: "left_source",
      target: spec,
      animated: false,
      style: { stroke: C.knowledge, strokeWidth: 1.5, opacity: 0.3, strokeDasharray: "6 3" },
    })
  ),

  // Specialists -> Arbitrator + Advisor
  makeEdge("dispute_spec", "arbitrator", undefined, C.arbitrator),
  makeEdge("scam_spec", "arbitrator", undefined, C.arbitrator),
  makeEdge("fraud_spec", "arbitrator", undefined, C.arbitrator),
  makeEdge("dispute_spec", "advisor", undefined, C.arbitrator),
  makeEdge("scam_spec", "advisor", undefined, C.arbitrator),
  makeEdge("fraud_spec", "advisor", undefined, C.arbitrator),


  // Synthesis -> Output
  makeEdge("arbitrator", "suggestion", undefined, C.output),
  makeEdge("advisor", "suggestion", undefined, C.output),

  // Conversation bar -> Transcript Stream (two arrows from active segment boundaries)
  {
    id: "bar_start-transcript",
    source: "conversation_bar",
    sourceHandle: "active_start",
    target: "transcript",
    animated: false,
    style: { stroke: C.input, strokeWidth: 2, opacity: 0.3 },
    labelStyle: { fill: "#94a3b8", fontSize: 9, fontWeight: 600 },
    labelBgStyle: { fill: "#1e293b", fillOpacity: 0.9 },
    labelBgPadding: [6, 3] as [number, number],
    labelBgBorderRadius: 4,
  },
  {
    id: "bar_end-transcript",
    source: "conversation_bar",
    sourceHandle: "active_end",
    target: "transcript",
    animated: false,
    style: { stroke: C.input, strokeWidth: 2, opacity: 0.3 },
    labelStyle: { fill: "#94a3b8", fontSize: 9, fontWeight: 600 },
    labelBgStyle: { fill: "#1e293b", fillOpacity: 0.9 },
    labelBgPadding: [6, 3] as [number, number],
    labelBgBorderRadius: 4,
  },

  // Copilot Suggestion -> CCP Desktop (right to left)
  {
    id: "suggestion-ccp_output",
    source: "suggestion",
    sourceHandle: "right",
    target: "ccp_output",
    targetHandle: "left",
    label: undefined,
    animated: false,
    style: { stroke: C.output, strokeWidth: 2, opacity: 0.3 },
    labelStyle: { fill: "#94a3b8", fontSize: 9, fontWeight: 600 },
    labelBgStyle: { fill: "#1e293b", fillOpacity: 0.9 },
    labelBgPadding: [6, 3] as [number, number],
    labelBgBorderRadius: 4,
  },
];

// --- Animation timeline ---
// Each step: which nodes are "running", which edges activate, duration ms

export type AnimationStep = {
  activeNodes: string[];
  activeEdges: string[];
  doneNodes?: string[];
  duration: number;
  label: string;
};

// Helper to build a pipeline run
const pipelineEdges = {
  stage1: ["orchestrator-triage", "orchestrator-retrieval"],
  stage1to2Specialists: [
    "triage-dispute_spec", "triage-scam_spec", "triage-fraud_spec",
    "retrieval-dispute_spec", "retrieval-scam_spec", "retrieval-fraud_spec",
  ],
  stage2Knowledge: knowledgeEdges,
  arbBase: [
    "dispute_spec-arbitrator", "scam_spec-arbitrator", "fraud_spec-arbitrator",
    "dispute_spec-advisor", "scam_spec-advisor", "fraud_spec-advisor",
  ],
  output: ["arbitrator-suggestion", "advisor-suggestion"],
};

function makeRun(runNum: number, turnLabel: string): AnimationStep[] {
  const r = `Run ${runNum}`;

  return [
    {
      activeNodes: ["conversation_bar", "transcript"],
      activeEdges: ["bar_start-transcript", "bar_end-transcript"],
      duration: 1500,
      label: runNum === 1
        ? `${r} \u2014 CCP triggers copilot for ${turnLabel}`
        : `${r} \u2014 Copilot auto-processes ${turnLabel}`,
    },
    {
      activeNodes: ["orchestrator"],
      activeEdges: ["transcript-orchestrator"],
      doneNodes: ["transcript"],
      duration: 1200,
      label: `${r} \u2014 Orchestrator decides which agents to trigger`,
    },
    {
      activeNodes: ["triage", "retrieval"],
      activeEdges: pipelineEdges.stage1,
      doneNodes: ["orchestrator"],
      duration: 1800,
      label: `${r} \u2014 Stage 1: Extraction + Retrieval in parallel`,
    },
    {
      activeNodes: ["dispute_spec", "scam_spec", "fraud_spec"],
      activeEdges: pipelineEdges.stage1to2Specialists,
      doneNodes: ["triage", "retrieval"],
      duration: 1500,
      label: `${r} \u2014 Stage 2: Specialists receive triage + retrieval results`,
    },
    {
      activeNodes: ["dispute_spec", "scam_spec", "fraud_spec", "knowledge_db"],
      activeEdges: pipelineEdges.stage2Knowledge,
      doneNodes: ["triage", "retrieval"],
      duration: 1800,
      label: `${r} \u2014 Stage 2: Specialists consult Knowledge Database for policy-aware assessment`,
    },
    {
      activeNodes: ["arbitrator", "advisor"],
      activeEdges: pipelineEdges.arbBase,
      doneNodes: ["dispute_spec", "scam_spec", "fraud_spec", "knowledge_db"],
      duration: 1800,
      label: `${r} \u2014 Stage 3: Arbitrator scores hypotheses; Advisor plans questions`,
    },
    {
      activeNodes: ["suggestion"],
      activeEdges: pipelineEdges.output,
      doneNodes: ["arbitrator", "advisor"],
      duration: 1200,
      label: `${r} \u2014 Copilot assembles suggestion for ${turnLabel}`,
    },
    {
      activeNodes: ["ccp_output"],
      activeEdges: ["suggestion-ccp_output"],
      doneNodes: ["suggestion"],
      duration: 1500,
      label: `${r} complete \u2014 Suggestion delivered to CCP`,
    },
  ];
}

export const animationSteps: AnimationStep[] = [
  // ── Conversation streams in ──
  {
    activeNodes: ["conversation_bar"],
    activeEdges: [],
    duration: 1500,
    label: "Live call begins \u2014 conversation streaming in...",
  },
  {
    activeNodes: ["conversation_bar"],
    activeEdges: [],
    duration: 1500,
    label: "Conversation continues \u2014 more turns arriving...",
  },
  {
    activeNodes: ["conversation_bar"],
    activeEdges: [],
    duration: 1500,
    label: "Conversation continues \u2014 more turns arriving...",
  },
  {
    activeNodes: ["conversation_bar"],
    activeEdges: [],
    duration: 2000,
    label: "CCP triggers copilot \u2014 it will run continuously from here",
  },

  // ── Run 1: Turn 1-10 ──
  ...makeRun(1, "Turn 1-10"),

  // ── More conversation streams during gap ──
  {
    activeNodes: ["conversation_bar"],
    activeEdges: [],
    duration: 1500,
    label: "Copilot auto-picks up new conversation queued during Run 1",
  },

  // ── Run 2: Turn 11-20 ──
  ...makeRun(2, "Turn 11-20"),

  // ── More conversation streams ──
  {
    activeNodes: ["conversation_bar"],
    activeEdges: [],
    duration: 1500,
    label: "Copilot auto-picks up new conversation queued during Run 2",
  },

  // ── Run 3: Turn 21-30 ──
  ...makeRun(3, "Turn 21-30"),
];
