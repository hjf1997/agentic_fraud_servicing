import React from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";

/**
 * Invisible anchor node positioned where the conversation progress bar is.
 * Only renders handles so edges can connect the bar to the React Flow graph.
 */
const BarAnchor: React.FC<NodeProps> = ({ data }) => {
  const color = (data as { color?: string }).color || "#f59e0b";
  return (
    <div style={{ width: 80, height: 1, position: "relative" }}>
      <Handle
        type="source"
        position={Position.Bottom}
        style={{ background: color, width: 8, height: 8, border: "none" }}
      />
      <Handle
        type="target"
        position={Position.Top}
        style={{ background: color, width: 8, height: 8, border: "none", top: -4 }}
      />
    </div>
  );
};

export default BarAnchor;
