import React from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";

/**
 * Conversation progress bar rendered as a React Flow node.
 *
 * Uses persistent named segments that transition between states
 * (queuing → active → processed) in-place via CSS class changes,
 * so the visual change is a smooth color shift, not a redraw.
 */

export type ConversationBarNodeData = {
  stepIdx: number;
};

type SegmentState = "processed" | "active" | "queuing" | "hidden";

type Segment = {
  id: string;
  label: string;
  state: SegmentState;
  width: number; // percentage of the track
};

type BarPhase = {
  segments: Segment[];
};

// 28 steps total:
// 0-1: conversation streams in
// 2-9: Run 1 (Turn 1-10) — r2 grows as queuing during pipeline
// 10: gap — more conversation streams
// 11-18: Run 2 (Turn 11-20) — r3 grows as queuing during pipeline
// 19: gap — more conversation streams
// 20-27: Run 3 (Turn 21-30) — no more queuing, call ends

type PhaseEntry = [
  r1State: SegmentState, r1Width: number,
  r2State: SegmentState, r2Width: number,
  r3State: SegmentState, r3Width: number,
];

const phases: PhaseEntry[] = [
  // 0-3: conversation streams in, then CCP triggers
  ["queuing", 8, "hidden", 0, "hidden", 0],      // 0: call begins
  ["queuing", 16, "hidden", 0, "hidden", 0],     // 1: more turns
  ["queuing", 24, "hidden", 0, "hidden", 0],     // 2: more turns
  ["queuing", 28, "hidden", 0, "hidden", 0],     // 3: CCP triggers

  // 4-10: Run 1 pipeline (7 steps, r1 active, r2 grows as queuing)
  ["active", 28, "hidden", 0, "hidden", 0],      // 4: transcript
  ["active", 28, "queuing", 4, "hidden", 0],      // 5: orchestrator
  ["active", 28, "queuing", 10, "hidden", 0],     // 6: phase 1
  ["active", 28, "queuing", 16, "hidden", 0],     // 7: phase 2
  ["active", 28, "queuing", 22, "hidden", 0],     // 8: arb + advisor
  ["active", 28, "queuing", 26, "hidden", 0],     // 9: suggestion
  ["processed", 28, "queuing", 28, "hidden", 0],  // 10: delivered

  // 11: gap — more conversation streams in
  ["processed", 28, "queuing", 30, "hidden", 0],

  // 12-18: Run 2 pipeline (7 steps, r2 active, r3 grows as queuing)
  ["processed", 28, "active", 30, "hidden", 0],      // 12: transcript
  ["processed", 28, "active", 30, "queuing", 3],      // 13: orchestrator
  ["processed", 28, "active", 30, "queuing", 7],      // 14: phase 1
  ["processed", 28, "active", 30, "queuing", 11],     // 15: phase 2
  ["processed", 28, "active", 30, "queuing", 15],     // 16: arb + advisor
  ["processed", 28, "active", 30, "queuing", 18],     // 17: suggestion
  ["processed", 28, "processed", 30, "queuing", 20],  // 18: delivered

  // 19: gap — more conversation streams in
  ["processed", 28, "processed", 30, "queuing", 42],

  // 20-26: Run 3 pipeline (7 steps, r3 active, fills to end)
  ["processed", 28, "processed", 30, "active", 42],    // 20: transcript
  ["processed", 28, "processed", 30, "active", 42],    // 21: orchestrator
  ["processed", 28, "processed", 30, "active", 42],    // 22: phase 1
  ["processed", 28, "processed", 30, "active", 42],    // 23: phase 2
  ["processed", 28, "processed", 30, "active", 42],    // 24: arb + advisor
  ["processed", 28, "processed", 30, "active", 42],    // 25: suggestion
  ["processed", 28, "processed", 30, "processed", 42], // 26: delivered — call complete
];

function barPhaseForStep(stepIdx: number): BarPhase {
  const p = phases[stepIdx] || phases[phases.length - 1];
  return { segments: [
    { id: "r1", label: "Turn 1-10", state: p[0], width: p[1] },
    { id: "r2", label: "Turn 11-20", state: p[2], width: p[3] },
    { id: "r3", label: "Turn 21-30", state: p[4], width: p[5] },
  ]};
}

/** Compute pixel offset for the start of the active segment within the track. */
function activeStartPx(segments: Segment[]): number {
  const trackLeft = 76; // CM icon (64) + gap (12)
  const trackWidth = 708;
  let offsetPct = 0;
  for (const seg of segments) {
    if (seg.state === "active") break;
    offsetPct += seg.width;
  }
  return trackLeft + (offsetPct / 100) * trackWidth + 6;
}

/** Compute pixel offset for the end of the active segment within the track. */
function activeEndPx(segments: Segment[]): number {
  const trackLeft = 76;
  const trackWidth = 708;
  let offsetPct = 0;
  let found = false;
  for (const seg of segments) {
    offsetPct += seg.width;
    if (seg.state === "active") { found = true; break; }
  }
  if (!found) return trackLeft + (offsetPct / 100) * trackWidth;
  return trackLeft + (offsetPct / 100) * trackWidth - 6;
}

const stateClass: Record<SegmentState, string> = {
  processed: "bar-section bar-processed",
  active: "bar-section bar-active",
  queuing: "bar-section bar-queuing",
  hidden: "bar-section bar-hidden",
};

const triggerSteps = new Set([3, 4]); // CCP triggers only once at the start

const ConversationBarNode: React.FC<NodeProps> = ({ data }) => {
  const d = data as unknown as ConversationBarNodeData;
  const phase = barPhaseForStep(d.stepIdx);
  const hasActive = phase.segments.some((s) => s.state === "active");
  const showTrigger = triggerSteps.has(d.stepIdx);

  return (
    <div style={{ width: 860 }}>
      {/* Header row */}
      <div className="conversation-bar-header">
        <div className="conversation-bar-title">
          <span className="conversation-bar-dot" />
          Live Conversation Timeline
        </div>
        <div className="conversation-bar-speakers">
          <span className="speaker-badge speaker-cm">CM</span>
          <span className="speaker-divider">/</span>
          <span className="speaker-badge speaker-ccp">CCP</span>
        </div>
      </div>

      {/* Bar with icons on each side */}
      <div className="conversation-bar-row">
        {/* CM icon */}
        <div className="bar-participant bar-participant-cm">
          <span className="bar-participant-icon">{"\uD83D\uDC64"}</span>
          <span className="bar-participant-name">Cardmember</span>
        </div>

        {/* Segment bar */}
        <div className="conversation-bar-track">
          {phase.segments.map((seg, i) => {
            // For processed segments: merge visually — only show label on the first one
            const isFirstProcessed =
              seg.state === "processed" &&
              (i === 0 || phase.segments[i - 1].state !== "processed");
            const showLabel =
              seg.state === "active" ||
              seg.state === "queuing" ||
              isFirstProcessed;

            let label = seg.label;
            if (seg.state === "processed") label = "Processed";
            if (seg.state === "queuing") label = "New conversation";

            return (
              <div
                key={seg.id}
                className={stateClass[seg.state]}
                style={{ width: `${seg.width}%` }}
              >
                {seg.state !== "hidden" && showLabel && seg.width >= 6 && (
                  <span className="bar-section-label">{label}</span>
                )}
              </div>
            );
          })}
        </div>

        {/* CCP icon */}
        <div className="bar-participant bar-participant-ccp" style={{ position: "relative" }}>
          <span className="bar-participant-icon">{"\uD83C\uDFA7"}</span>
          <span className="bar-participant-name">CCP</span>
          {showTrigger && (
            <div className="ccp-trigger-badge">
              Trigger Copilot
            </div>
          )}
        </div>
      </div>

      {/* Source handles at start and end of active segment */}
      {hasActive && (
        <>
          <Handle
            type="source"
            position={Position.Bottom}
            id="active_start"
            style={{
              background: "#006FCF",
              width: 8,
              height: 8,
              border: "none",
              left: activeStartPx(phase.segments),
            }}
          />
          <Handle
            type="source"
            position={Position.Bottom}
            id="active_end"
            style={{
              background: "#006FCF",
              width: 8,
              height: 8,
              border: "none",
              left: activeEndPx(phase.segments),
            }}
          />
        </>
      )}
    </div>
  );
};

export default ConversationBarNode;
