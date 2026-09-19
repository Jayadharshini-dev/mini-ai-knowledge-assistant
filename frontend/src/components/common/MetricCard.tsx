import React from "react";

interface MetricCardProps {
  label: string;
  value: string | number;
  subtext?: string;
  icon?: React.ReactNode;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  subtext,
  icon,
}) => {
  return (
    <div className="bg-white border border-slate-200 rounded-lg p-3.5 shadow-sm flex items-center justify-between">
      <div>
        <span className="text-[11px] uppercase font-mono font-medium text-slate-500 tracking-wider">
          {label}
        </span>
        <div className="text-xl font-bold font-mono text-slate-900 mt-1">
          {value}
        </div>
        {subtext && (
          <span className="text-xs text-slate-500 mt-0.5 block">{subtext}</span>
        )}
      </div>
      {icon && <div className="text-slate-400 bg-slate-50 p-2.5 rounded-md border border-slate-100">{icon}</div>}
    </div>
  );
};
