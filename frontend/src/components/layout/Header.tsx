import React from "react";
import { MessageSquare, Library, Activity, BookOpen } from "lucide-react";

export type ActiveTab = "chat" | "kb" | "trace";

interface HeaderProps {
  activeTab: ActiveTab;
  setActiveTab: (tab: ActiveTab) => void;
  kbState: "ready" | "empty" | "indexing" | "error";
  isBackendConnected: boolean;
  traceCount?: number;
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  setActiveTab,
  kbState,
  isBackendConnected,
  traceCount = 0,
}) => {
  return (
    <header className="bg-white border-b border-slate-200 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-14">
          {/* Subtle Branding */}
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-md bg-slate-900 text-white flex items-center justify-center font-medium shadow-sm">
              <BookOpen className="w-4 h-4" />
            </div>
            <div className="flex items-baseline gap-2">
              <span className="font-semibold text-sm tracking-tight text-slate-900">
                Mini AKA
              </span>
              <span className="hidden sm:inline-block text-xs text-slate-500 font-normal border-l border-slate-200 pl-2">
                Document Assistant
              </span>
            </div>
          </div>

          {/* Functional Navigation Tabs */}
          <nav className="flex items-center gap-1 sm:gap-1.5" aria-label="Main Navigation">
            <button
              onClick={() => setActiveTab("chat")}
              aria-current={activeTab === "chat" ? "page" : undefined}
              aria-label="Inquiry Screen"
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors focus-visible:ring-2 focus-visible:ring-slate-400 focus-visible:outline-none ${
                activeTab === "chat"
                  ? "bg-slate-100 text-slate-900 font-semibold"
                  : "text-slate-600 hover:text-slate-900 hover:bg-slate-50"
              }`}
            >
              <MessageSquare className="w-3.5 h-3.5 text-slate-500" />
              <span>Inquiry</span>
            </button>

            <button
              onClick={() => setActiveTab("kb")}
              aria-current={activeTab === "kb" ? "page" : undefined}
              aria-label="Document Library Screen"
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors focus-visible:ring-2 focus-visible:ring-slate-400 focus-visible:outline-none ${
                activeTab === "kb"
                  ? "bg-slate-100 text-slate-900 font-semibold"
                  : "text-slate-600 hover:text-slate-900 hover:bg-slate-50"
              }`}
            >
              <Library className="w-3.5 h-3.5 text-slate-500" />
              <span>Library</span>
            </button>

            <button
              onClick={() => setActiveTab("trace")}
              aria-current={activeTab === "trace" ? "page" : undefined}
              aria-label={`Trace Screen ${traceCount > 0 ? `(${traceCount} events)` : ""}`}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors focus-visible:ring-2 focus-visible:ring-slate-400 focus-visible:outline-none ${
                activeTab === "trace"
                  ? "bg-slate-100 text-slate-900 font-semibold"
                  : "text-slate-600 hover:text-slate-900 hover:bg-slate-50"
              }`}
            >
              <Activity className="w-3.5 h-3.5 text-slate-500" />
              <span className="hidden sm:inline">Trace</span>
              {traceCount > 0 && (
                <span className="ml-1 px-1.5 py-0.2 bg-slate-200 text-slate-800 text-[10px] rounded-full font-mono font-medium">
                  {traceCount}
                </span>
              )}
            </button>
          </nav>


          {/* Simple Connectivity & KB Status */}
          <div className="flex items-center gap-2 text-xs text-slate-500">
            <span
              className={`w-2 h-2 rounded-full ${
                !isBackendConnected
                  ? "bg-rose-500"
                  : kbState === "ready"
                  ? "bg-emerald-500"
                  : "bg-amber-500"
              }`}
            />
            <span className="text-[11px] font-medium text-slate-600 hidden sm:inline">
              {!isBackendConnected ? "Offline" : kbState === "ready" ? "Ready" : "No Documents"}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
};

