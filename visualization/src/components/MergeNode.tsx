import React from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";

export type MergeNodeData = {
  color: string;
  status?: "idle" | "running" | "done";
};

/**
 * Small circular merge point that joins multiple edges into one visual path.
 */
const MergeNode: React.FC<NodeProps> = ({ data }) => {
  const d = data as unknown as MergeNodeData;
  const status = d.status || "idle";
  const isActive = status === "running" || status === "done";

  return (
    <div
      style={{
        width: 10,
        height: 10,
        borderRadius: "50%",
        background: isActive ? d.color : `${d.color}44`,
        transition: "all 0.5s ease",
      }}
    >
      <Handle
        type="target"
        position={Position.Right}
        id="right_target"
        style={{ background: "transparent", border: "none", width: 1, height: 1 }}
      />
      <Handle
        type="source"
        position={Position.Bottom}
        style={{ background: "transparent", border: "none", width: 1, height: 1 }}
      />
      <Handle
        type="source"
        position={Position.Left}
        id="left_source"
        style={{ background: "transparent", border: "none", width: 1, height: 1 }}
      />
    </div>
  );
};

export default MergeNode;
