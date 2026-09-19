import React from "react";

interface BadgeProps {
  variant?: "default" | "success" | "warning" | "danger" | "mono" | "primary";
  children: React.ReactNode;
  size?: "sm" | "md";
}

export const Badge: React.FC<BadgeProps> = ({
  variant = "default",
  children,
  size = "md",
}) => {
  const getStyle = () => {
    switch (variant) {
      case "success":
        return "bg-emerald-100 text-emerald-800 border-emerald-200";
      case "warning":
        return "bg-amber-100 text-amber-800 border-amber-200";
      case "danger":
        return "bg-rose-100 text-rose-800 border-rose-200";
      case "primary":
        return "bg-blue-100 text-blue-800 border-blue-200";
      case "mono":
        return "bg-slate-100 text-slate-800 border-slate-300 font-mono";
      default:
        return "bg-slate-100 text-slate-700 border-slate-200";
    }
  };

  const sizeStyle = size === "sm" ? "px-1.5 py-0.5 text-[10px]" : "px-2.5 py-0.5 text-xs";

  return (
    <span
      className={`inline-flex items-center font-medium border rounded-md ${sizeStyle} ${getStyle()}`}
    >
      {children}
    </span>
  );
};
