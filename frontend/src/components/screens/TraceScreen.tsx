import React from "react";
import { Activity, ShieldCheck } from "lucide-react";

export const TraceScreen: React.FC = () => {
  return (
    <div className="space-y-6 max-w-6xl mx-auto w-full my-auto py-8">
      <div className="bg-white border border-slate-200 rounded-2xl p-10 text-center shadow-xs space-y-4">
        <div className="w-11 h-11 rounded-xl bg-slate-100 border border-slate-200 text-slate-700 flex items-center justify-center mx-auto shadow-2xs">
          <Activity className="w-5 h-5" />
        </div>
        <div className="space-y-1">
          <h2 className="text-base font-semibold text-slate-900 tracking-tight flex items-center justify-center gap-1.5">
            <span>Retrieval Trace & Provenance Verification</span>
            <ShieldCheck className="w-4 h-4 text-emerald-600 inline" />
          </h2>
          <p className="text-xs text-slate-500 max-w-md mx-auto leading-relaxed">
            Execution timelines, retrieval steps, verified source passages, and citation evidence will populate here automatically after submitting an inquiry.
          </p>
        </div>
      </div>
    </div>
  );
};

