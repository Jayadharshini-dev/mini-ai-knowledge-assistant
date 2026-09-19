import React from "react";
import { Activity } from "lucide-react";

export const TraceScreen: React.FC = () => {
  return (
    <div className="space-y-6 max-w-4xl mx-auto w-full my-auto py-8">
      <div className="bg-white border border-slate-200/90 rounded-2xl p-10 text-center shadow-sm space-y-3">
        <div className="w-12 h-12 rounded-2xl bg-indigo-50 border border-indigo-100 text-indigo-600 flex items-center justify-center mx-auto shadow-sm">
          <Activity className="w-6 h-6" />
        </div>
        <h2 className="text-base font-semibold text-slate-900 tracking-tight">
          Retrieval Trace & Sources
        </h2>
        <p className="text-xs text-slate-500 max-w-md mx-auto leading-relaxed">
          Retrieval details, verified source passages, and execution timings will appear here after you ask a question.
        </p>
      </div>
    </div>
  );
};
