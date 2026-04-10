import React, { useState, useRef, useEffect } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";

export type AgentNodeData = {
  label: string;
  subtitle?: string;
  icon: string;
  color: string;
  phase?: string;
  compact?: boolean;
  status?: "idle" | "running" | "done";
  onDataChange?: (newData: { label: string; subtitle?: string }) => void;
  onDelete?: () => void;
};

const AgentNode: React.FC<NodeProps> = ({ data }) => {
  const d = data as unknown as AgentNodeData;
  const status = d.status || "idle";

  const [editing, setEditing] = useState(false);
  const [hovered, setHovered] = useState(false);
  const [editLabel, setEditLabel] = useState(d.label);
  const [editSubtitle, setEditSubtitle] = useState(d.subtitle || "");
  const labelRef = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Sync external data changes when not editing
  useEffect(() => {
    if (!editing) {
      setEditLabel(d.label);
      setEditSubtitle(d.subtitle || "");
    }
  }, [d.label, d.subtitle, editing]);

  const commitEdit = () => {
    setEditing(false);
    if (d.onDataChange) {
      d.onDataChange({
        label: editLabel,
        subtitle: editSubtitle || undefined,
      });
    }
  };

  // Only commit when focus leaves the entire node, not between inputs
  const handleBlur = (e: React.FocusEvent) => {
    const related = e.relatedTarget as HTMLElement | null;
    if (containerRef.current && related && containerRef.current.contains(related)) {
      return; // focus moved to sibling input, don't commit
    }
    commitEdit();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      commitEdit();
    } else if (e.key === "Escape") {
      setEditing(false);
      setEditLabel(d.label);
      setEditSubtitle(d.subtitle || "");
    }
  };

  const borderColor =
    status === "running"
      ? d.color
      : status === "done"
      ? `${d.color}88`
      : `${d.color}44`;

  const shadowColor =
    status === "running" ? `${d.color}40` : "transparent";

  return (
    <div
      style={{
        background: "#ffffff",
        border: `2px solid ${borderColor}`,
        borderRadius: d.compact ? 8 : 12,
        padding: d.compact ? "6px 12px" : "14px 18px",
        minWidth: d.compact ? 140 : 170,
        boxShadow:
          status === "running"
            ? `0 0 20px ${shadowColor}, 0 0 40px ${shadowColor}`
            : "0 2px 8px rgba(0,0,0,0.08)",
        transition: "all 0.5s ease",
        opacity: status === "idle" ? 0.6 : 1,
        position: "relative",
      }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      onDoubleClick={(e) => {
        e.stopPropagation();
        setEditing(true);
        setTimeout(() => labelRef.current?.focus(), 0);
      }}
    >
      {/* Delete button */}
      {hovered && !editing && d.onDelete && (
        <button
          onClick={(e) => {
            e.stopPropagation();
            d.onDelete!();
          }}
          style={{
            position: "absolute",
            top: -8,
            right: -8,
            width: 20,
            height: 20,
            borderRadius: "50%",
            background: "#ef4444",
            border: "2px solid #ffffff",
            color: "#fff",
            fontSize: 11,
            fontWeight: 700,
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            lineHeight: 1,
            padding: 0,
            zIndex: 10,
          }}
          title="Delete node"
        >
          &times;
        </button>
      )}

      <Handle
        type="target"
        position={Position.Top}
        style={{ background: d.color, width: 8, height: 8, border: "none" }}
      />
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <span style={{ fontSize: 22 }}>{d.icon}</span>
        <div ref={containerRef} style={{ flex: 1, minWidth: 0 }}>
          {editing ? (
            <>
              <input
                ref={labelRef}
                value={editLabel}
                onChange={(e) => setEditLabel(e.target.value)}
                onBlur={handleBlur}
                onKeyDown={handleKeyDown}
                style={{
                  fontSize: 13,
                  fontWeight: 600,
                  color: "#00175A",
                  background: "#F7F8F9",
                  border: "1px solid #006FCF",
                  borderRadius: 4,
                  padding: "2px 6px",
                  width: "100%",
                  outline: "none",
                }}
              />
              <input
                value={editSubtitle}
                onChange={(e) => setEditSubtitle(e.target.value)}
                onBlur={handleBlur}
                onKeyDown={handleKeyDown}
                placeholder="subtitle"
                style={{
                  fontSize: 10,
                  color: "#53565A",
                  background: "#F7F8F9",
                  border: "1px solid #E0E0E0",
                  borderRadius: 4,
                  padding: "2px 6px",
                  width: "100%",
                  marginTop: 3,
                  outline: "none",
                }}
              />
            </>
          ) : (
            <>
              <div
                style={{
                  fontSize: 13,
                  fontWeight: 600,
                  color: "#00175A",
                  letterSpacing: 0.3,
                }}
              >
                {d.label}
              </div>
              {d.subtitle && (
                <div style={{ fontSize: 10, color: "#53565A", marginTop: 2 }}>
                  {d.subtitle}
                </div>
              )}
            </>
          )}
        </div>
        {status === "running" && (
          <div
            style={{
              width: 8,
              height: 8,
              borderRadius: "50%",
              background: d.color,
              marginLeft: "auto",
              animation: "pulse 1.2s ease-in-out infinite",
            }}
          />
        )}
        {status === "done" && (
          <span
            style={{
              marginLeft: "auto",
              fontSize: 14,
              color: "#008000",
            }}
          >
            &#10003;
          </span>
        )}
      </div>
      {d.phase && (
        <div
          style={{
            fontSize: 9,
            color: d.color,
            textTransform: "uppercase",
            letterSpacing: 1,
            marginTop: 6,
            fontWeight: 600,
          }}
        >
          {d.phase}
        </div>
      )}
      <Handle
        type="source"
        position={Position.Bottom}
        style={{ background: d.color, width: 8, height: 8, border: "none" }}
      />
      <Handle
        type="source"
        position={Position.Right}
        id="right"
        style={{ background: d.color, width: 8, height: 8, border: "none" }}
      />
      <Handle
        type="target"
        position={Position.Left}
        id="left"
        style={{ background: d.color, width: 8, height: 8, border: "none" }}
      />
    </div>
  );
};

export default AgentNode;
