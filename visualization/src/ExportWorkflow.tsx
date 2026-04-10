import React, { useState, useEffect, useCallback, useRef } from "react";
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  BackgroundVariant,
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
import {
  initialNodes,
  initialEdges,
  animationSteps,
  type AnimationStep,
} from "./flowData";
import type { AgentNodeData } from "./components/AgentNode";

const nodeTypes = { agent: AgentNode, group: GroupLabel, conversationBar: ConversationBarNode };

function applyStep(
  step: AnimationStep,
  baseNodes: Node[],
  baseEdges: Edge[]
): { nodes: Node[]; edges: Edge[] } {
  const activeSet = new Set(step.activeNodes);
  const doneSet = new Set(step.doneNodes || []);
  const activeEdgeSet = new Set(step.activeEdges);

  const nodes = baseNodes.map((n) => {
    if (n.type !== "agent") return n;
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

function ExportInner() {
  const [stepIdx, setStepIdx] = useState(0);
  const [playing, setPlaying] = useState(true);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [currentNodes, setCurrentNodes] = useState<Node[]>(initialNodes);
  const [currentEdges, setCurrentEdges] = useState<Edge[]>(initialEdges);

  const updateNodeInternals = useUpdateNodeInternals();
  useEffect(() => {
    updateNodeInternals("conversation_bar");
  }, [stepIdx, updateNodeInternals]);

  const allDone = useRef<Set<string>>(new Set());

  useEffect(() => {
    if (stepIdx === 0) {
      allDone.current = new Set();
    }
  }, [stepIdx]);

  const step = animationSteps[stepIdx];
  if (step.doneNodes) {
    step.doneNodes.forEach((id) => allDone.current.add(id));
  }

  const enrichedStep: AnimationStep = {
    ...step,
    doneNodes: Array.from(allDone.current),
  };

  const nodesWithCallbacks = currentNodes.map((n) => ({
    ...n,
    data: {
      ...n.data,
      ...(n.type === "conversationBar" ? { stepIdx } : {}),
    },
  }));

  const { nodes, edges } = applyStep(enrichedStep, nodesWithCallbacks, currentEdges);

  const onNodesChange = useCallback((changes: NodeChange[]) => {
    setCurrentNodes((nds) => applyNodeChanges(changes, nds) as Node[]);
  }, []);

  const onEdgesChange = useCallback((changes: EdgeChange[]) => {
    setCurrentEdges((eds) => applyEdgeChanges(changes, eds) as Edge[]);
  }, []);

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

  return (
    <div
      style={{
        width: "100vw",
        height: "100vh",
        display: "flex",
        flexDirection: "column",
        position: "relative",
        background: "#ffffff",
      }}
    >
      {/* Title */}
      <div
        style={{
          padding: "14px 32px 0",
          flexShrink: 0,
        }}
      >
        <div
          style={{
            fontSize: 22,
            fontWeight: 700,
            color: "#00175A",
            letterSpacing: 0.5,
          }}
        >
          Copilot Agent Workflow
          <span
            style={{
              fontSize: 13,
              color: "#53565A",
              fontWeight: 400,
              marginLeft: 14,
            }}
          >
            AMEX Fraud Servicing
          </span>
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
          nodesDraggable={false}
          nodesConnectable={false}
          edgesReconnectable={false}
          deleteKeyCode={null}
          proOptions={{ hideAttribution: true }}
          minZoom={0.4}
          maxZoom={1.5}
          panOnDrag={false}
          zoomOnScroll={false}
          zoomOnPinch={false}
          zoomOnDoubleClick={false}
          preventScrolling={false}
        >
          <Background
            variant={BackgroundVariant.Dots}
            color="#C8C9C7"
            gap={20}
            size={1}
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
          padding: "40px 32px 18px",
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
            fontSize: 14,
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
        `}
      </style>
    </div>
  );
}

export default function ExportWorkflow() {
  return (
    <ReactFlowProvider>
      <ExportInner />
    </ReactFlowProvider>
  );
}
