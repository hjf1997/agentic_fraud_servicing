import React, { useState, useEffect, useCallback, useRef } from "react";
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  BackgroundVariant,
  Controls,
  applyNodeChanges,
  applyEdgeChanges,
  useUpdateNodeInternals,
  type Node,
  type Edge,
  type NodeChange,
  type EdgeChange,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import AgentNode from "./components/AgentNode";
import GroupLabel from "./components/GroupLabel";
import ConversationBarNode from "./components/ConversationBarNode";
import MergeNode from "./components/MergeNode";
import {
  initialNodes,
  initialEdges,
  animationSteps,
  type AnimationStep,
} from "./flowData";
import type { AgentNodeData } from "./components/AgentNode";

const nodeTypes = { agent: AgentNode, group: GroupLabel, conversationBar: ConversationBarNode, merge: MergeNode };

function applyStep(
  step: AnimationStep,
  baseNodes: Node[],
  baseEdges: Edge[]
): { nodes: Node[]; edges: Edge[] } {
  const activeSet = new Set(step.activeNodes);
  const doneSet = new Set(step.doneNodes || []);
  const activeEdgeSet = new Set(step.activeEdges);

  const nodes = baseNodes.map((n) => {
    if (n.type !== "agent" && n.type !== "merge") return n;
    const d = n.data as unknown as AgentNodeData;
    let status: "idle" | "running" | "done" = "idle";
    if (activeSet.has(n.id)) status = "running";
    else if (doneSet.has(n.id)) status = "done";
    return { ...n, data: { ...d, status } };
  });

  const edges = baseEdges.map((e) => {
    const isActive = activeEdgeSet.has(e.id);
    return {
      ...e,
      animated: isActive,
      style: {
        ...e.style,
        opacity: isActive ? 1 : 0.2,
        strokeWidth: isActive ? 2.5 : 1.5,
      },
    };
  });

  return { nodes, edges };
}

function AppInner() {
  const [stepIdx, setStepIdx] = useState(0);
  const [playing, setPlaying] = useState(true);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Store current node positions and data (updated by dragging / editing)
  const [currentNodes, setCurrentNodes] = useState<Node[]>(initialNodes);
  // Store current edges (updated by deletion)
  const [currentEdges, setCurrentEdges] = useState<Edge[]>(initialEdges);

  // Save status feedback
  const [saveStatus, setSaveStatus] = useState<"idle" | "saving" | "saved" | "error">("idle");

  // Force React Flow to recalculate handle positions when stepIdx changes
  const updateNodeInternals = useUpdateNodeInternals();
  useEffect(() => {
    updateNodeInternals("conversation_bar");
  }, [stepIdx, updateNodeInternals]);

  // Track cumulative done nodes across steps
  const allDone = useRef<Set<string>>(new Set());

  // Reset cumulative state when restarting
  useEffect(() => {
    if (stepIdx === 0) {
      allDone.current = new Set();
    }
  }, [stepIdx]);

  // Build cumulative done set
  const step = animationSteps[stepIdx];
  if (step.doneNodes) {
    step.doneNodes.forEach((id) => allDone.current.add(id));
  }

  const enrichedStep: AnimationStep = {
    ...step,
    doneNodes: Array.from(allDone.current),
  };

  // Inject stepIdx into conversation bar node and editing callbacks into others
  const nodesWithCallbacks = currentNodes.map((n) => ({
    ...n,
    data: {
      ...n.data,
      ...(n.type === "conversationBar" ? { stepIdx } : {}),
      onDataChange: (newData: Record<string, unknown>) => {
        setCurrentNodes((nds) =>
          nds.map((node) =>
            node.id === n.id
              ? { ...node, data: { ...node.data, ...newData } }
              : node
          )
        );
      },
      onDelete: () => {
        setCurrentNodes((nds) => nds.filter((node) => node.id !== n.id));
        setCurrentEdges((eds) =>
          eds.filter((e) => e.source !== n.id && e.target !== n.id)
        );
      },
    },
  }));

  // Apply animation status on top of current (possibly dragged) positions
  const { nodes, edges } = applyStep(
    enrichedStep,
    nodesWithCallbacks,
    currentEdges
  );

  // Handle node drag and other position changes
  const onNodesChange = useCallback((changes: NodeChange[]) => {
    setCurrentNodes((nds) => applyNodeChanges(changes, nds) as Node[]);
  }, []);

  // Handle edge changes (selection, deletion)
  const onEdgesChange = useCallback((changes: EdgeChange[]) => {
    setCurrentEdges((eds) => applyEdgeChanges(changes, eds) as Edge[]);
  }, []);

  // Delete selected nodes/edges on Delete/Backspace key
  const handleKeyDown = useCallback(
    (event: React.KeyboardEvent) => {
      if (event.key === "Delete" || event.key === "Backspace") {
        // Delete selected edges
        setCurrentEdges((eds) => {
          const selected = eds.filter((e) => e.selected);
          if (selected.length === 0) return eds;
          return eds.filter((e) => !e.selected);
        });
        // Delete selected nodes and their connected edges
        setCurrentNodes((nds) => {
          const selected = nds.filter((n) => n.selected);
          if (selected.length === 0) return nds;
          const deletedIds = new Set(selected.map((n) => n.id));
          setCurrentEdges((eds) =>
            eds.filter(
              (e) => !deletedIds.has(e.source) && !deletedIds.has(e.target)
            )
          );
          return nds.filter((n) => !n.selected);
        });
      }
    },
    []
  );

  // Save to server
  const handleSave = useCallback(async () => {
    setSaveStatus("saving");
    try {
      const resp = await fetch("/api/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ nodes: currentNodes, edges: currentEdges }),
      });
      if (!resp.ok) throw new Error(`Server returned ${resp.status}`);
      setSaveStatus("saved");
      setTimeout(() => setSaveStatus("idle"), 2000);
    } catch (err) {
      console.error("Save failed:", err);
      setSaveStatus("error");
      setTimeout(() => setSaveStatus("idle"), 3000);
    }
  }, [currentNodes, currentEdges]);

  // Auto-advance
  useEffect(() => {
    if (!playing) return;
    timerRef.current = setTimeout(() => {
      setStepIdx((i) => (i + 1) % animationSteps.length);
    }, animationSteps[stepIdx].duration);
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [stepIdx, playing]);

  const togglePlay = useCallback(() => setPlaying((p) => !p), []);
  const goNext = useCallback(
    () => setStepIdx((i) => (i + 1) % animationSteps.length),
    []
  );
  const goPrev = useCallback(
    () =>
      setStepIdx(
        (i) => (i - 1 + animationSteps.length) % animationSteps.length
      ),
    []
  );
  const restart = useCallback(() => {
    setStepIdx(0);
    setPlaying(true);
  }, []);

  const saveButtonLabel =
    saveStatus === "saving"
      ? "Saving..."
      : saveStatus === "saved"
      ? "Saved!"
      : saveStatus === "error"
      ? "Error!"
      : "Save";

  const saveButtonBg =
    saveStatus === "saved"
      ? "#008000"
      : saveStatus === "error"
      ? "#CF291D"
      : "#006FCF";

  return (
    <div
      style={{ width: "100vw", height: "100vh", display: "flex", flexDirection: "column", position: "relative" }}
      onKeyDown={handleKeyDown}
      tabIndex={0}
    >
      {/* Title row */}
      <div
        style={{
          padding: "12px 24px 0",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          flexShrink: 0,
        }}
      >
        <div
          style={{
            fontSize: 18,
            fontWeight: 700,
            color: "#00175A",
            letterSpacing: 0.5,
          }}
        >
          Copilot Agent Workflow
          <span
            style={{
              fontSize: 11,
              color: "#53565A",
              fontWeight: 400,
              marginLeft: 12,
            }}
          >
            AMEX Fraud Servicing
          </span>
        </div>
        <div style={{ display: "flex", gap: 10 }}>
          <a
            href="#/demo"
            style={{
              background: "#F7F8F9",
              border: "1px solid #E0E0E0",
              borderRadius: 8,
              color: "#00175A",
              fontSize: 13,
              fontWeight: 600,
              padding: "8px 20px",
              textDecoration: "none",
              letterSpacing: 0.3,
            }}
          >
            Live Demo
          </a>
          <a
            href="#/agent"
            style={{
              background: "#F7F8F9",
              border: "1px solid #E0E0E0",
              borderRadius: 8,
              color: "#00175A",
              fontSize: 13,
              fontWeight: 600,
              padding: "8px 20px",
              textDecoration: "none",
              letterSpacing: 0.3,
            }}
          >
            Agents
          </a>
          <a
            href="#/layers"
            style={{
              background: "#F7F8F9",
              border: "1px solid #E0E0E0",
              borderRadius: 8,
              color: "#00175A",
              fontSize: 13,
              fontWeight: 600,
              padding: "8px 20px",
              textDecoration: "none",
              letterSpacing: 0.3,
            }}
          >
            Architecture
          </a>
          <button
            onClick={handleSave}
            disabled={saveStatus === "saving"}
            style={{
              background: saveButtonBg,
              border: "none",
              borderRadius: 8,
              color: "#fff",
              fontSize: 13,
              fontWeight: 600,
              padding: "8px 20px",
              cursor: saveStatus === "saving" ? "wait" : "pointer",
              transition: "all 0.3s ease",
              letterSpacing: 0.3,
            }}
          >
            {saveButtonLabel}
          </button>
        </div>
      </div>

      {/* React Flow canvas */}
      <div style={{ flex: 1, position: "relative" }}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          fitView
          fitViewOptions={{ padding: 0.15 }}
          nodesDraggable={true}
          nodesConnectable={false}
          edgesReconnectable={false}
          deleteKeyCode={null}
          proOptions={{ hideAttribution: true }}
          minZoom={0.4}
          maxZoom={1.5}
        >
          <Background
            variant={BackgroundVariant.Dots}
            color="#C8C9C7"
            gap={20}
            size={1}
          />
          <Controls
            showInteractive={false}
            style={{ background: "#ffffff", borderColor: "#E0E0E0" }}
          />
        </ReactFlow>
      </div>

      {/* Bottom bar: step label + controls */}
      <div
        style={{
          position: "absolute",
          bottom: 0,
          left: 0,
          right: 0,
          background: "linear-gradient(transparent, rgba(255,255,255,0.95) 30%)",
          padding: "40px 24px 20px",
          display: "flex",
          alignItems: "center",
          gap: 16,
        }}
      >
        {/* Progress dots */}
        <div style={{ display: "flex", gap: 6 }}>
          {animationSteps.map((_, i) => (
            <div
              key={i}
              onClick={() => setStepIdx(i)}
              style={{
                width: i === stepIdx ? 24 : 8,
                height: 8,
                borderRadius: 4,
                background: i === stepIdx ? "#006FCF" : "#C8C9C7",
                cursor: "pointer",
                transition: "all 0.3s ease",
              }}
            />
          ))}
        </div>

        {/* Step label */}
        <div
          style={{
            flex: 1,
            fontSize: 13,
            color: "#00175A",
            fontWeight: 500,
          }}
        >
          <span style={{ color: "#006FCF", marginRight: 8 }}>
            {stepIdx + 1}/{animationSteps.length}
          </span>
          {animationSteps[stepIdx].label}
        </div>

        {/* Controls */}
        <div style={{ display: "flex", gap: 8 }}>
          {[
            { label: "\u23EE", action: restart, title: "Restart" },
            { label: "\u25C0", action: goPrev, title: "Previous" },
            {
              label: playing ? "\u23F8" : "\u25B6",
              action: togglePlay,
              title: playing ? "Pause" : "Play",
            },
            { label: "\u25B6", action: goNext, title: "Next" },
          ].map((btn, i) => (
            <button
              key={i}
              onClick={btn.action}
              title={btn.title}
              style={{
                background: "#F7F8F9",
                border: "1px solid #E0E0E0",
                borderRadius: 6,
                color: "#00175A",
                fontSize: 16,
                width: 36,
                height: 36,
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              {btn.label}
            </button>
          ))}
        </div>
      </div>

      {/* Pulse animation */}
      <style>
        {`
          @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.4; transform: scale(1.5); }
          }
          .react-flow__edge-path {
            transition: opacity 0.4s ease, stroke-width 0.4s ease;
          }
          .react-flow__edge.selected .react-flow__edge-path {
            stroke: #006FCF !important;
            stroke-width: 3px !important;
            opacity: 1 !important;
          }
        `}
      </style>
    </div>
  );
}

export default function App() {
  return (
    <ReactFlowProvider>
      <AppInner />
    </ReactFlowProvider>
  );
}
