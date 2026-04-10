/**
 * Lightweight save server for the React Flow visualization editor.
 * Receives edited nodes/edges via POST /api/save and writes them
 * back to src/flowData.ts, preserving the TypeScript structure.
 */

const express = require("express");
const fs = require("fs");
const path = require("path");

const app = express();
const PORT = 3457;
const FLOW_DATA_PATH = path.join(__dirname, "src", "flowData.ts");

app.use(express.json({ limit: "2mb" }));

/**
 * Map a hex color value back to a color constant key.
 * Falls back to a quoted hex string if no match is found.
 */
function colorToKey(hex) {
  const map = {
    "#f59e0b": "C.input",
    "#8b5cf6": "C.orchestrator",
    "#3b82f6": "C.phase1",
    "#10b981": "C.specialist",
    "#f43f5e": "C.arbitrator",
    "#06b6d4": "C.advisor",
    "#a78bfa": "C.output",
    "#64748b": "C.group",
  };
  return map[hex] || `"${hex}"`;
}

/** Escape a string for use inside a TypeScript string literal. */
function esc(s) {
  if (s == null) return "";
  return s.replace(/\\/g, "\\\\").replace(/"/g, '\\"');
}

/**
 * Generate the full flowData.ts content from nodes and edges.
 * Closely mirrors the original hand-written structure.
 */
function generateFlowDataTs(nodes, edges, existingAnimationSteps) {
  const agentNodes = nodes.filter((n) => n.type === "agent");
  const groupNodes = nodes.filter((n) => n.type === "group");

  // Clean node IDs that still exist (for animation step cleanup)
  const existingNodeIds = new Set(nodes.map((n) => n.id));
  const existingEdgeIds = new Set(
    edges.map((e) => `${e.source}-${e.target}`)
  );

  let out = `import type { Node, Edge } from "@xyflow/react";
import type { AgentNodeData } from "./components/AgentNode";
import type { GroupLabelData } from "./components/GroupLabel";

// Colors by role
const C = {
  input: "#f59e0b", // amber
  orchestrator: "#8b5cf6", // violet
  phase1: "#3b82f6", // blue
  specialist: "#10b981", // emerald
  arbitrator: "#f43f5e", // rose
  advisor: "#06b6d4", // cyan
  output: "#a78bfa", // light violet
  group: "#64748b", // slate
};

type AgentNode = Node<AgentNodeData, "agent">;
type GroupNode = Node<GroupLabelData, "group">;

// --- Nodes ---

const agentNodes: AgentNode[] = [\n`;

  for (const n of agentNodes) {
    const d = n.data;
    out += `  {\n`;
    out += `    id: "${esc(n.id)}",\n`;
    out += `    type: "agent",\n`;
    out += `    position: { x: ${Math.round(n.position.x)}, y: ${Math.round(n.position.y)} },\n`;
    out += `    data: {\n`;
    out += `      label: "${esc(d.label)}",\n`;
    if (d.subtitle) {
      out += `      subtitle: "${esc(d.subtitle)}",\n`;
    }
    out += `      icon: "${esc(d.icon)}",\n`;
    out += `      color: ${colorToKey(d.color)},\n`;
    if (d.phase) {
      out += `      phase: "${esc(d.phase)}",\n`;
    }
    out += `    },\n`;
    out += `  },\n`;
  }

  out += `];\n\nconst groupNodes: GroupNode[] = [\n`;

  for (const n of groupNodes) {
    const d = n.data;
    out += `  {\n`;
    out += `    id: "${esc(n.id)}",\n`;
    out += `    type: "group",\n`;
    out += `    position: { x: ${Math.round(n.position.x)}, y: ${Math.round(n.position.y)} },\n`;
    out += `    data: {\n`;
    out += `      label: "${esc(d.label)}",\n`;
    out += `      color: ${colorToKey(d.color)},\n`;
    out += `      width: ${d.width},\n`;
    out += `      height: ${d.height},\n`;
    out += `    },\n`;
    out += `    draggable: false,\n`;
    out += `    selectable: false,\n`;
    out += `  },\n`;
  }

  out += `];\n\nexport const initialNodes: Node[] = [\n  ...groupNodes,\n  ...agentNodes,\n];\n\n`;

  // --- Edges ---
  out += `// --- Edges ---\n\n`;
  out += `const makeEdge = (\n  source: string,\n  target: string,\n  label?: string,\n  color?: string,\n): Edge => ({\n`;
  out += `  id: \`\${source}-\${target}\`,\n`;
  out += `  source,\n  target,\n  label,\n  animated: false,\n`;
  out += `  style: { stroke: color || C.group, strokeWidth: 2, opacity: 0.3 },\n`;
  out += `  labelStyle: { fill: "#94a3b8", fontSize: 9, fontWeight: 600 },\n`;
  out += `  labelBgStyle: { fill: "#1e293b", fillOpacity: 0.9 },\n`;
  out += `  labelBgPadding: [6, 3] as [number, number],\n`;
  out += `  labelBgBorderRadius: 4,\n`;
  out += `});\n\n`;

  out += `export const initialEdges: Edge[] = [\n`;
  for (const e of edges) {
    const lbl = e.label ? `"${esc(String(e.label))}"` : "undefined";
    const clr = e.style && e.style.stroke ? colorToKey(e.style.stroke) : "undefined";
    out += `  makeEdge("${esc(e.source)}", "${esc(e.target)}", ${lbl}, ${clr}),\n`;
  }
  out += `];\n\n`;

  // --- Animation timeline ---
  out += `// --- Animation timeline ---\n`;
  out += `// Each step: which nodes are "running", which edges activate, duration ms\n\n`;
  out += `export type AnimationStep = {\n`;
  out += `  activeNodes: string[];\n`;
  out += `  activeEdges: string[];\n`;
  out += `  doneNodes?: string[];\n`;
  out += `  duration: number;\n`;
  out += `  label: string;\n`;
  out += `};\n\n`;

  // Clean animation steps: remove references to deleted nodes/edges
  const cleanedSteps = existingAnimationSteps.map((step) => ({
    ...step,
    activeNodes: step.activeNodes.filter((id) => existingNodeIds.has(id)),
    activeEdges: step.activeEdges.filter((id) => existingEdgeIds.has(id)),
    doneNodes: step.doneNodes
      ? step.doneNodes.filter((id) => existingNodeIds.has(id))
      : undefined,
  }));

  out += `export const animationSteps: AnimationStep[] = [\n`;
  for (const s of cleanedSteps) {
    out += `  {\n`;
    out += `    activeNodes: [${s.activeNodes.map((id) => `"${esc(id)}"`).join(", ")}],\n`;
    out += `    activeEdges: [\n`;
    for (const ae of s.activeEdges) {
      out += `      "${esc(ae)}",\n`;
    }
    out += `    ],\n`;
    if (s.doneNodes && s.doneNodes.length > 0) {
      out += `    doneNodes: [${s.doneNodes.map((id) => `"${esc(id)}"`).join(", ")}],\n`;
    }
    out += `    duration: ${s.duration},\n`;
    out += `    label: "${esc(s.label)}",\n`;
    out += `  },\n`;
  }
  out += `];\n`;

  return out;
}

app.post("/api/save", (req, res) => {
  try {
    const { nodes, edges } = req.body;
    if (!nodes || !edges) {
      return res.status(400).json({ error: "Missing nodes or edges" });
    }

    // Read existing animation steps from the current file to preserve them
    // (we parse them from the request if provided, otherwise re-read from file)
    let existingAnimationSteps;
    try {
      const currentContent = fs.readFileSync(FLOW_DATA_PATH, "utf-8");
      // Extract the animationSteps array using a simple regex approach
      const match = currentContent.match(
        /export const animationSteps: AnimationStep\[\] = (\[[\s\S]*?\n\]);/
      );
      if (match) {
        // Use Function constructor to evaluate the array (safe since it's our own file)
        existingAnimationSteps = new Function(`return ${match[1]}`)();
      }
    } catch {
      // Fallback: empty animation steps
    }

    if (!existingAnimationSteps) {
      existingAnimationSteps = [];
    }

    // Strip internal callbacks (onDataChange, onDelete) and status from node data
    const cleanNodes = nodes.map((n) => {
      const { onDataChange, onDelete, status, ...cleanData } = n.data;
      return { ...n, data: cleanData };
    });

    const content = generateFlowDataTs(cleanNodes, edges, existingAnimationSteps);
    fs.writeFileSync(FLOW_DATA_PATH, content, "utf-8");

    res.json({ ok: true });
  } catch (err) {
    console.error("Save failed:", err);
    res.status(500).json({ error: err.message });
  }
});

app.listen(PORT, () => {
  console.log(`Save server running on http://localhost:${PORT}`);
});
