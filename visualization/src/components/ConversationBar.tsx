import React from "react";

/**
 * Conversation progress bar showing the live call timeline as a
 * continuously growing bar. Three colored sections:
 *   - Processed (slate) — turns already consumed by previous copilot runs
 *   - Active (amber, glowing) — turns being consumed by the current run
 *   - Queuing (amber outline, growing) — new turns arriving during processing
 *
 * All widths are percentages driven by stepIdx, with CSS transitions
 * creating smooth growth animations.
 */

type BarState = {
  /** Width of processed section as % of full bar */
  processed: number;
  /** Width of active section as % of full bar */
  active: number;
  /** Width of queuing section as % of full bar */
  queuing: number;
  /** Label shown inside the active section */
  activeLabel: string;
};

/**
 * Maps animation step to bar widths. The bar grows over time:
 *
 *  Step 0: Transcript receives turns     → active appears
 *  Step 1: Orchestrator                  → queuing starts growing
 *  Step 2: Phase 1                       → queuing grows more
 *  Step 3: Sync                          → queuing grows more
 *  Step 4: Phase 2                       → queuing grows more
 *  Step 5: Decision & guidance           → queuing grows more
 *  Step 6: Suggestion assembled          → active → processed, queuing → active
 *  Step 7: Loop                          → new queuing appears, cycle ready
 */
function barStateForStep(stepIdx: number): BarState {
  switch (stepIdx) {
    case 0:
      return { processed: 40, active: 20, queuing: 0, activeLabel: "Turn 5-6" };
    case 1:
      return { processed: 40, active: 20, queuing: 5, activeLabel: "Turn 5-6" };
    case 2:
      return { processed: 40, active: 20, queuing: 10, activeLabel: "Turn 5-6" };
    case 3:
      return { processed: 40, active: 20, queuing: 14, activeLabel: "Turn 5-6" };
    case 4:
      return { processed: 40, active: 20, queuing: 18, activeLabel: "Turn 5-6" };
    case 5:
      return { processed: 40, active: 20, queuing: 22, activeLabel: "Turn 5-6" };
    case 6:
      // Active batch done → becomes processed; queuing → active
      return { processed: 60, active: 22, queuing: 0, activeLabel: "Turn 7-8" };
    case 7:
      // New queuing starts appearing for next cycle
      return { processed: 60, active: 22, queuing: 8, activeLabel: "Turn 7-8" };
    default:
      return { processed: 40, active: 20, queuing: 0, activeLabel: "Turn 5-6" };
  }
}

type ConversationBarProps = {
  stepIdx: number;
};

const ConversationBar: React.FC<ConversationBarProps> = ({ stepIdx }) => {
  const state = barStateForStep(stepIdx);

  return (
    <div className="conversation-bar-container">
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
        {/* CM icon — left side */}
        <div className="bar-participant bar-participant-cm">
          <span className="bar-participant-icon">{"\uD83D\uDC64"}</span>
          <span className="bar-participant-name">Cardmember</span>
        </div>

        {/* Growing bar */}
        <div className="conversation-bar-track">
          {/* Processed section */}
          <div
            className="bar-section bar-processed"
            style={{ width: `${state.processed}%` }}
          >
            <span className="bar-section-label">Processed</span>
          </div>

          {/* Boundary marker */}
          {state.active > 0 && <div className="bar-boundary" />}

          {/* Active section */}
          <div
            className="bar-section bar-active"
            style={{ width: `${state.active}%` }}
          >
            <span className="bar-section-label">{state.activeLabel}</span>
          </div>

          {/* Queuing section */}
          {state.queuing > 0 && (
            <div
              className="bar-section bar-queuing"
              style={{ width: `${state.queuing}%` }}
            >
              <span className="bar-section-label">New turns</span>
            </div>
          )}
        </div>

        {/* CCP icon — right side */}
        <div className="bar-participant bar-participant-ccp">
          <span className="bar-participant-icon">{"\uD83C\uDFA7"}</span>
          <span className="bar-participant-name">CCP</span>
        </div>
      </div>
    </div>
  );
};

export default ConversationBar;
