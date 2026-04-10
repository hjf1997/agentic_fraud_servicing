import React, { useState, useRef, useEffect } from "react";
import { type NodeProps } from "@xyflow/react";

export type GroupLabelData = {
  label: string;
  color: string;
  width: number;
  height: number;
  onDataChange?: (newData: { label: string }) => void;
  onDelete?: () => void;
};

const GroupLabel: React.FC<NodeProps> = ({ data }) => {
  const d = data as unknown as GroupLabelData;

  const [editing, setEditing] = useState(false);
  const [hovered, setHovered] = useState(false);
  const [editLabel, setEditLabel] = useState(d.label);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!editing) {
      setEditLabel(d.label);
    }
  }, [d.label, editing]);

  const commitEdit = () => {
    setEditing(false);
    if (d.onDataChange) {
      d.onDataChange({ label: editLabel });
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      commitEdit();
    } else if (e.key === "Escape") {
      setEditing(false);
      setEditLabel(d.label);
    }
  };

  return (
    <div
      style={{
        width: d.width,
        height: d.height,
        border: `1px dashed ${d.color}33`,
        borderRadius: 16,
        background: `${d.color}08`,
        padding: "10px 16px",
        position: "relative",
      }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      onDoubleClick={(e) => {
        e.stopPropagation();
        setEditing(true);
        setTimeout(() => inputRef.current?.focus(), 0);
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
          title="Delete group"
        >
          &times;
        </button>
      )}

      <div
        style={{
          position: "absolute",
          top: -12,
          left: 16,
          background: "#ffffff",
          padding: "2px 10px",
          borderRadius: 4,
        }}
      >
        {editing ? (
          <input
            ref={inputRef}
            value={editLabel}
            onChange={(e) => setEditLabel(e.target.value)}
            onBlur={commitEdit}
            onKeyDown={handleKeyDown}
            style={{
              fontSize: 12,
              fontWeight: 700,
              color: d.color,
              textTransform: "uppercase",
              letterSpacing: 1.5,
              background: "#ffffff",
              border: "1px solid #006FCF",
              borderRadius: 4,
              padding: "2px 8px",
              outline: "none",
            }}
          />
        ) : (
          <span
            style={{
              fontSize: 12,
              fontWeight: 700,
              color: d.color,
              textTransform: "uppercase",
              letterSpacing: 1.5,
            }}
          >
            {d.label}
          </span>
        )}
      </div>
    </div>
  );
};

export default GroupLabel;
