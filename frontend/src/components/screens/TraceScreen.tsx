import React, { useState } from "react";
import {
  Activity,
  ShieldCheck,
  ChevronDown,
  ChevronRight,
  Clock,
} from "lucide-react";
import type { InquiryResult } from "../../types/api";
import type { TraceEvent } from "../../types/events";


interface TraceScreenProps {
  traceEvents?: TraceEvent[];
  inquiryResult?: InquiryResult | null;
}

export const TraceScreen: React.FC<TraceScreenProps> = ({
  traceEvents = [],
  inquiryResult,
}) => {
  const [expandedSeq, setExpandedSeq] = useState<number | null>(null);

  const toggleExpand = (seq: number) => {
    setExpandedSeq(expandedSeq === seq ? null : seq);
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "ok":
        return <span className="w-2 h-2 rounded-full bg-emerald-500" />;
      case "warn":
        return <span className="w-2 h-2 rounded-full bg-amber-500" />;
      case "error":
        return <span className="w-2 h-2 rounded-full bg-rose-500" />;
      default:
        return <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />;
    }
  };

  const latestEvent = traceEvents[traceEvents.length - 1];

  return (
    <div className="space-y-6 max-w-5xl mx-auto w-full py-4">
      {/* Header Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-slate-200">
        <div>
          <h2 className="text-lg font-semibold text-slate-900 tracking-tight flex items-center gap-2">
            <span>Retrieval Trace & Provenance</span>
            <ShieldCheck className="w-4.5 h-4.5 text-emerald-600" />
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Real-time pipeline execution sequence, relevance decisions, and timing metrics.
          </p>
        </div>

        {traceEvents.length > 0 && (
          <div className="flex items-center gap-3 text-xs font-mono bg-white border border-slate-200 px-3 py-1.5 rounded-xl shadow-2xs">
            <div className="flex items-center gap-1.5 text-slate-600">
              <Clock className="w-3.5 h-3.5 text-slate-400" />
              <span>{latestEvent?.t_ms ?? 0}ms</span>
            </div>
            <span className="text-slate-300">|</span>
            <div className="flex items-center gap-1.5">
              <span>{traceEvents.length} Events</span>
            </div>
          </div>
        )}
      </div>

      {/* Empty Trace State */}
      {traceEvents.length === 0 ? (
        <div className="bg-white border border-slate-200 rounded-2xl p-12 text-center shadow-2xs space-y-4 my-8">
          <div className="w-12 h-12 rounded-2xl bg-slate-100 border border-slate-200 text-slate-600 flex items-center justify-center mx-auto shadow-2xs">
            <Activity className="w-6 h-6" />
          </div>
          <div className="space-y-1">
            <h3 className="text-base font-semibold text-slate-900 tracking-tight">
              No Active Inquiry Trace
            </h3>
            <p className="text-xs text-slate-500 max-w-md mx-auto leading-relaxed">
              Submit a question in the Inquiry tab to inspect real-time vector retrieval steps, relevance threshold decisions, and timing metrics.
            </p>
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          {/* Active Query Context Card */}
          {inquiryResult?.question && (
            <div className="bg-slate-100/80 border border-slate-200 rounded-xl p-3.5 text-xs flex items-center justify-between">
              <div className="truncate pr-2">
                <span className="font-mono text-[10px] text-slate-400 uppercase tracking-wider block">
                  Active Inquiry Target
                </span>
                <span className="font-medium text-slate-900 truncate block">
                  "{inquiryResult.question}"
                </span>
              </div>
              <span className="font-mono text-[11px] text-slate-500 shrink-0">
                {inquiryResult.isAbstained
                  ? "Abstained"
                  : inquiryResult.answer
                  ? "Completed"
                  : "Streaming"}
              </span>
            </div>
          )}

          {/* Trace Event Timeline */}
          <div
            role="region"
            aria-label="Retrieval Trace Timeline"
            className="bg-white border border-slate-200 rounded-2xl overflow-hidden shadow-2xs"
          >
            <div className="px-4 py-3 bg-slate-50/70 border-b border-slate-200 flex items-center justify-between text-[11px] font-semibold text-slate-500 uppercase tracking-wider font-mono">
              <div className="flex items-center gap-6">
                <span>Seq</span>
                <span>Event Type</span>
              </div>
              <div className="flex items-center gap-8">
                <span>Status / Label</span>
                <span>Timing</span>
              </div>
            </div>

            <div className="divide-y divide-slate-100 text-xs">
              {traceEvents.map((evt) => {
                const isExpanded = expandedSeq === evt.seq;
                const detailKeys = Object.keys(evt.detail || {});
                const isExpandable = detailKeys.length > 0;

                return (
                  <div key={evt.seq} className="hover:bg-slate-50/50 transition-colors">
                    <div
                      tabIndex={isExpandable ? 0 : undefined}
                      role={isExpandable ? "button" : undefined}
                      aria-expanded={isExpandable ? isExpanded : undefined}
                      aria-label={isExpandable ? `Trace step ${evt.seq} ${evt.type}: ${evt.label}. Click or press Enter to ${isExpanded ? "collapse" : "expand"} details.` : undefined}
                      onClick={() => isExpandable && toggleExpand(evt.seq)}
                      onKeyDown={(e) => {
                        if (isExpandable && (e.key === "Enter" || e.key === " ")) {
                          e.preventDefault();
                          toggleExpand(evt.seq);
                        }
                      }}
                      className={`px-4 py-3 flex items-center justify-between ${
                        isExpandable ? "cursor-pointer focus-visible:ring-2 focus-visible:ring-slate-400 focus-visible:outline-none" : ""
                      } ${isExpanded ? "bg-slate-50/80" : ""}`}
                    >
                      <div className="flex items-center gap-4 min-w-0">
                        <span className="font-mono text-slate-400 text-[11px] w-6">
                          #{evt.seq}
                        </span>
                        <div className="flex items-center gap-2 font-mono font-medium text-slate-900">
                          {getStatusBadge(evt.status)}
                          <span className="bg-slate-100 text-slate-800 px-2 py-0.5 rounded text-[11px] font-semibold border border-slate-200">
                            {evt.type}
                          </span>
                        </div>
                      </div>

                      <div className="flex items-center gap-4 text-right truncate pl-4">
                        <span className="text-slate-600 truncate max-w-xs font-normal">
                          {evt.label}
                        </span>
                        <span className="font-mono text-slate-400 text-[11px] w-14 shrink-0 text-right">
                          {evt.t_ms}ms
                        </span>
                        {detailKeys.length > 0 && (
                          <div className="text-slate-400">
                            {isExpanded ? (
                              <ChevronDown className="w-4 h-4" />
                            ) : (
                              <ChevronRight className="w-4 h-4" />
                            )}
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Expandable Detail Drawer */}
                    {isExpanded && detailKeys.length > 0 && (
                      <div className="px-12 py-3 bg-slate-900 text-slate-100 text-[11px] font-mono rounded-b-lg space-y-2 border-t border-slate-800">
                        <div className="text-slate-400 uppercase tracking-wider text-[10px]">
                          Event Details ({evt.type})
                        </div>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-1.5">
                          {Object.entries(evt.detail).map(([key, value]) => {
                            if (typeof value === "object" && value !== null) {
                              return (
                                <div key={key} className="col-span-full">
                                  <span className="text-slate-400">{key}:</span>{" "}
                                  <span className="text-emerald-300">
                                    {JSON.stringify(value, null, 2)}
                                  </span>
                                </div>
                              );
                            }
                            return (
                              <div key={key} className="flex justify-between border-b border-slate-800/80 pb-0.5">
                                <span className="text-slate-400">{key}</span>
                                <span className="text-emerald-300 font-semibold">{String(value)}</span>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};


