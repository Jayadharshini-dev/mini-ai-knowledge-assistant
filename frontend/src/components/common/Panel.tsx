import React from "react";

interface PanelProps {
  title?: string;
  subtitle?: string;
  headerAction?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}

export const Panel: React.FC<PanelProps> = ({
  title,
  subtitle,
  headerAction,
  children,
  className = "",
}) => {
  return (
    <div
      className={`bg-white border border-slate-200 rounded-lg shadow-sm flex flex-col overflow-hidden ${className}`}
    >
      {title && (
        <div className="border-b border-slate-100 bg-slate-50/50 px-4 py-3 flex items-center justify-between">
          <div>
            <h3 className="text-sm font-semibold text-slate-800 uppercase tracking-wider font-mono">
              {title}
            </h3>
            {subtitle && (
              <p className="text-xs text-slate-500 mt-0.5">{subtitle}</p>
            )}
          </div>
          {headerAction && <div>{headerAction}</div>}
        </div>
      )}
      <div className="p-4 flex-1">{children}</div>
    </div>
  );
};
