import React from "react";
import { MessageSquare, Library, Activity, Sparkles } from "lucide-react";

export type ActiveTab = "chat" | "kb" | "trace";

interface HeaderProps {
  activeTab: ActiveTab;
  setActiveTab: (tab: ActiveTab) => void;
  kbState: "ready" | "empty" | "indexing" | "error";
  isBackendConnected: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  setActiveTab,
  kbState,
  isBackendConnected,
}) => {
  return (
    <header className="bg-white/90 backdrop-blur-md border-b border-slate-200/80 sticky top-0 z-50">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-14">
          {/* Logo & Product Name */}
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-indigo-600 text-white flex items-center justify-center shadow-sm">
              <Sparkles className="w-4 h-4" />
            </div>
            <div>
              <span className="font-semibold text-sm tracking-tight text-slate-900">
                Mini AKA
              </span>
              <span className="hidden sm:inline-block text-xs text-slate-500 ml-2 font-normal border-l border-slate-200 pl-2">
                Document Assistant
              </span>
            </div>
          </div>

          {/* Clean Navigation Tabs */}
          <nav className="flex items-center gap-1 sm:gap-2">
            <button
              onClick={() => setActiveTab("chat")}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                activeTab === "chat"
                  ? "bg-slate-100 text-indigo-600 font-semibold"
                  : "text-slate-600 hover:text-slate-900 hover:bg-slate-50"
              }`}
            >
              <MessageSquare className="w-3.5 h-3.5" />
              <span>Inquiry</span>
            </button>

            <button
              onClick={() => setActiveTab("kb")}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                activeTab === "kb"
                  ? "bg-slate-100 text-indigo-600 font-semibold"
                  : "text-slate-600 hover:text-slate-900 hover:bg-slate-50"
              }`}
            >
              <Library className="w-3.5 h-3.5" />
              <span>Library</span>
            </button>

            <button
              onClick={() => setActiveTab("trace")}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                activeTab === "trace"
                  ? "bg-slate-100 text-indigo-600 font-semibold"
                  : "text-slate-600 hover:text-slate-900 hover:bg-slate-50"
              }`}
            >
              <Activity className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Trace</span>
            </button>
          </nav>

          {/* Simple Status Indicator */}
          <div className="flex items-center gap-1.5 text-xs text-slate-500">
            <span
              className={`w-2 h-2 rounded-full ${
                !isBackendConnected
                  ? "bg-rose-500"
                  : kbState === "ready"
                  ? "bg-emerald-500"
                  : "bg-amber-500"
              }`}
            />
            <span className="text-[11px] hidden md:inline font-medium">
              {!isBackendConnected ? "Offline" : kbState === "ready" ? "Ready" : "No Docs"}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
};
