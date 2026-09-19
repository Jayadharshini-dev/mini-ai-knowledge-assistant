import React from "react";

interface StatusIndicatorProps {
  status: "online" | "offline" | "connecting" | "ready" | "empty" | "indexing" | "error";
  label?: string;
  size?: "sm" | "md";
}

export const StatusIndicator: React.FC<StatusIndicatorProps> = ({
  status,
  label,
}) => {
  const getColors = () => {
    switch (status) {
      case "online":
      case "ready":
        return {
          bg: "bg-emerald-50 text-emerald-700 border-emerald-200",
          dot: "bg-emerald-500 animate-pulse",
        };
      case "connecting":
      case "indexing":
        return {
          bg: "bg-amber-50 text-amber-700 border-amber-200",
          dot: "bg-amber-500 animate-ping",
        };
      case "empty":
        return {
          bg: "bg-slate-100 text-slate-700 border-slate-200",
          dot: "bg-slate-400",
        };
      case "offline":
      case "error":
        return {
          bg: "bg-rose-50 text-rose-700 border-rose-200",
          dot: "bg-rose-500",
        };
    }
  };

  const colors = getColors();

  return (
    <div
      className={`inline-flex items-center gap-2 rounded-full border px-2.5 py-0.5 text-xs font-medium font-mono ${colors.bg}`}
    >
      <span className={`h-2 w-2 rounded-full ${colors.dot}`} />
      <span>{label || status.toUpperCase()}</span>
    </div>
  );
};
