import React, { useState } from "react";
import {
  Send,
  ArrowUpRight,
  BookOpen,
  FileText,
  Upload,
  Layers,
  AlertTriangle,
  AlertCircle,
  FileCheck,
  ChevronDown,
  ChevronUp,
  RotateCcw,
} from "lucide-react";
import type {
  DocumentRecord,
  InquiryResult,
  KnowledgeBaseStatus,
} from "../../types/api";
import type { AbstainedDetail, CompleteDetail, TraceEvent } from "../../types/events";
import { streamChat } from "../../services/api";

interface ChatScreenProps {
  status: KnowledgeBaseStatus | null;
  documents?: DocumentRecord[];
  traceEvents: TraceEvent[];
  setTraceEvents: React.Dispatch<React.SetStateAction<TraceEvent[]>>;
  inquiryResult: InquiryResult | null;
  setInquiryResult: React.Dispatch<React.SetStateAction<InquiryResult | null>>;
  onNavigateToDocs?: () => void;
  isBackendConnected?: boolean;
}

export const ChatScreen: React.FC<ChatScreenProps> = ({
  status,
  documents = [],
  setTraceEvents,
  inquiryResult,
  setInquiryResult,
  onNavigateToDocs,
  isBackendConnected = true,
}) => {
  const [questionInput, setQuestionInput] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showSubThresholdPassages, setShowSubThresholdPassages] = useState(false);

  const isKbReady = status?.state === "ready" && documents.length > 0;

  const handleSubmit = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const query = questionInput.trim();
    if (!query || isSubmitting || !isKbReady) return;

    setIsSubmitting(true);
    setInquiryResult(null);
    setTraceEvents([]);
    setShowSubThresholdPassages(false);

    let activeAbstained: AbstainedDetail | null = null;

    try {
      await streamChat(
        query,
        (evt) => {
          setTraceEvents((prev) => [...prev, evt]);
        },
        (completeDetail: CompleteDetail) => {
          setInquiryResult({
            question: query,
            answer: completeDetail.answer,
            citations: completeDetail.citations || [],
            chunks: completeDetail.chunks || [],
            isAbstained: completeDetail.abstained,
            isDegraded: Boolean(completeDetail.degraded),
            degradedReason: completeDetail.degraded,
            abstainMessage: activeAbstained?.message,
            traceEvents: [],
            elapsedMs: completeDetail.elapsed_ms,
          });
        },
        (abstainedDetail: AbstainedDetail) => {
          activeAbstained = abstainedDetail;
        },
        (error) => {
          setInquiryResult({
            question: query,
            answer: null,
            citations: [],
            chunks: activeAbstained?.chunks || [],
            isAbstained: activeAbstained !== null,
            isDegraded: false,
            abstainMessage: activeAbstained?.message,
            traceEvents: [],
            error,
          });
        }
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleReset = () => {
    setQuestionInput("");
    setInquiryResult(null);
    setTraceEvents([]);
    setIsSubmitting(false);
  };

  const evidenceChunks = inquiryResult?.chunks.filter((c) => c.above_threshold) || [];
  const subThresholdChunks = inquiryResult?.chunks.filter((c) => !c.above_threshold) || [];

  return (
    <div className="flex-1 max-w-7xl mx-auto w-full flex flex-col justify-between">
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8 flex-1 items-start">
        {/* Main Conversation Column (3 Cols on Desktop) */}
        <div className="lg:col-span-3 flex flex-col justify-between h-full min-h-[calc(100vh-12rem)] space-y-6">
          {/* Backend Connection Offline Alert Notice */}
          {!isBackendConnected ? (
            <div className="bg-rose-50 border border-rose-200 rounded-xl p-4 flex items-center gap-3 text-xs text-rose-800">
              <AlertCircle className="w-4 h-4 shrink-0 text-rose-600" />
              <div>
                <span className="font-semibold">Backend Connection Offline</span>
                <p className="text-rose-700 text-[11px] mt-0.5">
                  Unable to connect to assistant server. Please ensure the backend process is running on port 8000.
                </p>
              </div>
            </div>
          ) : (
            /* Empty Document Library Alert Notice */
            !isKbReady && (
              <div className="bg-slate-100/90 border border-slate-200 rounded-xl p-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 text-xs text-slate-700">
                <div className="flex items-center gap-2.5">
                  <div className="w-7 h-7 rounded-lg bg-amber-100 text-amber-800 flex items-center justify-center shrink-0">
                    <Upload className="w-3.5 h-3.5" />
                  </div>
                  <div>
                    <span className="font-semibold text-slate-900">Document library is empty</span>
                    <p className="text-slate-500 text-[11px] mt-0.5">
                      Upload PDF documents in the Library to enable grounded answers.
                    </p>
                  </div>
                </div>
                {onNavigateToDocs && (
                  <button
                    onClick={onNavigateToDocs}
                    className="inline-flex items-center gap-1 text-xs font-semibold text-slate-900 hover:text-indigo-600 shrink-0 focus-visible:ring-2 focus-visible:ring-slate-400 focus-visible:outline-none rounded"
                  >
                    <span>Go to Library</span>
                    <ArrowUpRight className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>
            )
          )}

          {/* Welcome / Initial Empty Conversation View */}
          {!isSubmitting && !inquiryResult && (
            <div className="flex-1 flex flex-col justify-center my-auto py-12">
              <div className="text-center space-y-3 max-w-lg mx-auto">
                <div className="w-11 h-11 rounded-xl bg-slate-100 border border-slate-200 text-slate-700 flex items-center justify-center mx-auto shadow-sm">
                  <BookOpen className="w-5 h-5" />
                </div>
                <h2 className="text-xl font-semibold text-slate-900 tracking-tight">
                  Ask anything about your documents
                </h2>
                <p className="text-xs text-slate-500 leading-relaxed max-w-md mx-auto">
                  Inquire across your document library. Answers are generated using verified source passages with precise citations.
                </p>
              </div>
            </div>
          )}

          {/* Active Query Streaming Progress View */}
          {isSubmitting && (
            <div className="flex-1 flex flex-col justify-center my-auto py-12 text-center space-y-4 max-w-md mx-auto">
              <div className="w-10 h-10 rounded-xl bg-slate-900 text-white flex items-center justify-center mx-auto shadow-sm">
                <div className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
              </div>
              <div>
                <h3 className="text-sm font-semibold text-slate-900">
                  Executing Inquiry
                </h3>
                <p className="text-xs text-slate-500 mt-1">
                  Searching document index & validating passage relevance...
                </p>
              </div>
            </div>
          )}

          {/* Inquiry Result Display View */}
          {inquiryResult && (
            <div className="flex-1 space-y-6 py-2">
              {/* Question Header */}
              <div className="bg-slate-100/70 border border-slate-200/80 rounded-2xl p-4 flex items-start justify-between gap-4">
                <div className="space-y-1">
                  <span className="text-[10px] uppercase font-mono font-semibold text-slate-500 tracking-wider">
                    Inquiry
                  </span>
                  <h2 className="text-base font-semibold text-slate-900">
                    {inquiryResult.question}
                  </h2>
                </div>
                <button
                  onClick={handleReset}
                  className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-white border border-slate-200 text-slate-600 hover:text-slate-900 text-xs font-medium shrink-0 transition-colors shadow-2xs"
                >
                  <RotateCcw className="w-3 h-3" />
                  <span>New Inquiry</span>
                </button>
              </div>

              {/* Error Alert */}
              {inquiryResult.error && (
                <div className="bg-rose-50 border border-rose-200 rounded-2xl p-5 space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2 text-rose-800 font-semibold text-xs">
                      <AlertCircle className="w-4 h-4 shrink-0" />
                      <span>Inquiry Error ({inquiryResult.error.code})</span>
                    </div>
                    {inquiryResult.error.retryable && (
                      <button
                        onClick={() => handleSubmit()}
                        disabled={isSubmitting || !isBackendConnected}
                        className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-rose-700 hover:bg-rose-800 text-white rounded-lg text-xs font-medium shadow-2xs transition-colors focus-visible:ring-2 focus-visible:ring-rose-500 focus-visible:outline-none disabled:opacity-50"
                      >
                        <RotateCcw className="w-3 h-3" />
                        <span>Retry Inquiry</span>
                      </button>
                    )}
                  </div>
                  <p className="text-xs text-rose-700 leading-relaxed">
                    {inquiryResult.error.message}
                  </p>
                </div>
              )}

              {/* Abstention Notice */}
              {inquiryResult.isAbstained && (
                <div className="bg-amber-50/80 border border-amber-200 rounded-2xl p-5 space-y-4">
                  <div className="flex items-start gap-3">
                    <div className="w-8 h-8 rounded-xl bg-amber-100 text-amber-800 flex items-center justify-center shrink-0">
                      <AlertTriangle className="w-4 h-4" />
                    </div>
                    <div className="space-y-1">
                      <h3 className="text-sm font-semibold text-amber-950">
                        Knowledge Base Abstention
                      </h3>
                      <p className="text-xs text-amber-900 leading-relaxed">
                        {inquiryResult.abstainMessage ||
                          "The knowledge base does not contain information relevant to this question."}
                      </p>
                    </div>
                  </div>

                  {/* Sub-threshold passages drawer if present */}
                  {subThresholdChunks.length > 0 && (
                    <div className="pt-2 border-t border-amber-200/60">
                      <button
                        onClick={() => setShowSubThresholdPassages(!showSubThresholdPassages)}
                        className="inline-flex items-center gap-1.5 text-xs font-medium text-amber-900 hover:underline"
                      >
                        <span>
                          {showSubThresholdPassages ? "Hide" : "Inspect"} sub-threshold passages ({subThresholdChunks.length})
                        </span>
                        {showSubThresholdPassages ? (
                          <ChevronUp className="w-3.5 h-3.5" />
                        ) : (
                          <ChevronDown className="w-3.5 h-3.5" />
                        )}
                      </button>

                      {showSubThresholdPassages && (
                        <div className="mt-3 space-y-2">
                          {subThresholdChunks.map((item, idx) => (
                            <div
                              key={idx}
                              className="bg-white border border-amber-200 rounded-xl p-3 text-xs space-y-1.5"
                            >
                              <div className="flex items-center justify-between text-[11px] text-slate-500">
                                <span className="font-medium text-slate-800">
                                  {item.chunk.document} (Page {item.chunk.page})
                                </span>
                                <span className="font-mono text-amber-800 font-medium">
                                  similarity (cosine): {item.score.toFixed(3)}
                                </span>
                              </div>
                              <p className="text-slate-600 italic leading-relaxed text-[11px] bg-slate-50 p-2 rounded-lg border border-slate-100">
                                "{item.chunk.text}"
                              </p>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* Degraded / No-Provider Notice when answer is null & not abstained */}
              {!inquiryResult.answer && !inquiryResult.isAbstained && !inquiryResult.error && (
                <div className="bg-slate-100 border border-slate-200 rounded-2xl p-5 space-y-2">
                  <div className="flex items-center gap-2 text-slate-800 font-semibold text-xs">
                    <Layers className="w-4 h-4 shrink-0 text-slate-600" />
                    <span>Evidence-Only Response</span>
                  </div>
                  <p className="text-xs text-slate-600 leading-relaxed">
                    No answer was generated. Language-model generation is currently unavailable. The relevant document passages are shown below.
                  </p>
                </div>
              )}

              {/* Grounded Answer */}
              {inquiryResult.answer && (
                <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-2xs space-y-4">
                  <div className="flex items-center justify-between pb-3 border-b border-slate-100">
                    <div className="flex items-center gap-2">
                      <FileCheck className="w-4 h-4 text-emerald-600" />
                      <h3 className="text-xs font-semibold text-slate-900 uppercase tracking-wider font-mono">
                        Grounded Answer
                      </h3>
                    </div>
                    {inquiryResult.elapsedMs && (
                      <span className="text-[11px] font-mono text-slate-400">
                        {inquiryResult.elapsedMs}ms
                      </span>
                    )}
                  </div>
                  <div className="text-sm text-slate-900 leading-relaxed whitespace-pre-wrap">
                    {inquiryResult.answer}
                  </div>
                </div>
              )}

              {/* Verified Sources / Evidence (Only above_threshold chunks) */}
              {evidenceChunks.length > 0 && (
                <div className="space-y-3 pt-2">
                  <div className="flex items-center justify-between">
                    <h3 className="text-xs font-semibold text-slate-900 uppercase tracking-wider font-mono">
                      Verified Sources ({evidenceChunks.length})
                    </h3>
                    <span className="text-[11px] text-slate-500">
                      Grounded in uploaded document passages
                    </span>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {evidenceChunks.map((retrieved, idx) => {
                      const citation = inquiryResult.citations.find(
                        (c) => c.chunk_id === retrieved.chunk.chunk_id
                      );

                      return (
                        <div
                          key={retrieved.chunk.chunk_id || idx}
                          className="bg-white border border-slate-200 rounded-xl p-3.5 shadow-2xs space-y-2 hover:border-slate-300 transition-colors"
                        >
                          <div className="flex items-start justify-between gap-2 border-b border-slate-100 pb-2">
                            <div className="flex items-center gap-2 truncate">
                              <FileText className="w-3.5 h-3.5 text-slate-600 shrink-0" />
                              <span className="font-medium text-xs text-slate-900 truncate" title={retrieved.chunk.document}>
                                {retrieved.chunk.document}
                              </span>
                            </div>
                            <span className="text-[10px] font-mono bg-slate-100 text-slate-700 px-2 py-0.5 rounded-full font-medium shrink-0">
                              similarity (cosine): {retrieved.score.toFixed(3)}
                            </span>
                          </div>

                          <div className="flex items-center gap-3 text-[11px] text-slate-500 font-mono">
                            <span>Page {retrieved.chunk.page}</span>
                            <span>•</span>
                            <span>Section #{retrieved.chunk.chunk_index}</span>
                            {citation && (
                              <>
                                <span>•</span>
                                <span className="text-slate-900 font-semibold">
                                  Citation [{citation.n}]
                                </span>
                              </>
                            )}
                          </div>

                          <p className="text-xs text-slate-600 leading-relaxed bg-slate-50 p-2.5 rounded-lg border border-slate-100 line-clamp-4">
                            "{retrieved.chunk.text}"
                          </p>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Question Composer */}
          <div className="w-full pt-4 pb-2">
            <form
              onSubmit={handleSubmit}
              className="relative bg-white border border-slate-200 rounded-2xl shadow-xs focus-within:border-slate-400 focus-within:shadow-sm transition-all p-2.5"
            >
              <label htmlFor="chat-inquiry-input" className="sr-only">
                Ask a question about your documents
              </label>
              <textarea
                id="chat-inquiry-input"
                rows={2}
                value={questionInput}
                onChange={(e) => setQuestionInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit();
                  }
                }}
                placeholder={
                  !isBackendConnected
                    ? "Backend offline. Reconnect server to ask questions..."
                    : isKbReady
                    ? "Ask a question about your documents..."
                    : "Upload documents to start asking questions..."
                }
                disabled={!isBackendConnected || !isKbReady || isSubmitting}
                className="w-full resize-none bg-transparent px-3 py-1.5 text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed"
              />
              <div className="flex items-center justify-between pt-2 px-2 border-t border-slate-100">
                <span className="text-[11px] text-slate-400 flex items-center gap-1">
                  <Layers className="w-3 h-3 text-slate-400" />
                  Grounded in verified document passages
                </span>
                <button
                  type="submit"
                  aria-label="Submit Question"
                  disabled={!isBackendConnected || !isKbReady || !questionInput.trim() || isSubmitting}
                  className="w-8 h-8 rounded-xl bg-slate-900 text-white flex items-center justify-center hover:bg-slate-800 disabled:opacity-30 disabled:hover:bg-slate-900 transition-colors shadow-xs focus-visible:ring-2 focus-visible:ring-slate-400 focus-visible:outline-none"
                >
                  <Send className="w-3.5 h-3.5" />
                </button>
              </div>
            </form>
          </div>
        </div>

        {/* Desktop Contextual Document Rail (Secondary Info Area) */}
        <div className="hidden lg:block lg:col-span-1 bg-white border border-slate-200 rounded-2xl p-4 shadow-xs sticky top-20">
          <div className="flex items-center justify-between pb-3 border-b border-slate-100">
            <div className="flex items-center gap-2">
              <FileText className="w-4 h-4 text-slate-600" />
              <h3 className="text-xs font-semibold text-slate-900">
                Active Library Context
              </h3>
            </div>
            <span className="text-[10px] font-medium font-mono px-2 py-0.5 rounded-full bg-slate-100 text-slate-600 border border-slate-200">
              {documents.length} {documents.length === 1 ? "File" : "Files"}
            </span>
          </div>

          <div className="mt-3 space-y-2.5 max-h-[calc(100vh-18rem)] overflow-y-auto pr-1">
            {documents.length === 0 ? (
              <div className="py-6 text-center text-slate-400">
                <p className="text-xs text-slate-500">No documents in library.</p>
                {onNavigateToDocs && (
                  <button
                    onClick={onNavigateToDocs}
                    className="mt-2 text-xs font-medium text-slate-900 underline hover:text-indigo-600"
                  >
                    Upload files
                  </button>
                )}
              </div>
            ) : (
              documents.map((doc) => (
                <div
                  key={doc.doc_id}
                  className="p-2.5 rounded-xl border border-slate-100 bg-slate-50/50 space-y-1 hover:border-slate-200 transition-colors"
                >
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-medium text-slate-800 truncate max-w-[140px]" title={doc.filename}>
                      {doc.filename}
                    </span>
                    <span
                      className={`text-[10px] font-medium capitalize ${
                        doc.status === "indexed" ? "text-emerald-700" : "text-rose-700"
                      }`}
                    >
                      {doc.status}
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-[11px] text-slate-500">
                    <span>{doc.pages} {doc.pages === 1 ? "page" : "pages"}</span>
                    <span>{doc.chunks} sections</span>
                  </div>
                </div>
              ))
            )}
          </div>

          {documents.length > 0 && onNavigateToDocs && (
            <div className="pt-3 mt-3 border-t border-slate-100">
              <button
                onClick={onNavigateToDocs}
                className="w-full text-center text-xs text-slate-600 hover:text-slate-900 font-medium py-1"
              >
                Manage Document Library →
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};



