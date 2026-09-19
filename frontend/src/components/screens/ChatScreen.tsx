import React, { useState } from "react";
import { Send, ArrowUpRight, BookOpen, FileText, Upload, Layers } from "lucide-react";
import type { DocumentRecord, KnowledgeBaseStatus } from "../../types/api";

interface ChatScreenProps {
  status: KnowledgeBaseStatus | null;
  documents?: DocumentRecord[];
  onNavigateToDocs?: () => void;
}

export const ChatScreen: React.FC<ChatScreenProps> = ({
  status,
  documents = [],
  onNavigateToDocs,
}) => {
  const [question, setQuestion] = useState("");
  const isKbReady = status?.state === "ready" && documents.length > 0;

  return (
    <div className="flex-1 max-w-7xl mx-auto w-full flex flex-col justify-between">
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8 flex-1 items-start">
        {/* Main Conversation Column (3 Cols on Desktop) */}
        <div className="lg:col-span-3 flex flex-col justify-between h-full min-h-[calc(100vh-12rem)]">
          {/* Empty Document Library Alert Notice */}
          {!isKbReady && (
            <div className="mb-6 bg-slate-100/90 border border-slate-200 rounded-xl p-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 text-xs text-slate-700">
              <div className="flex items-center gap-2.5">
                <div className="w-7 h-7 rounded-lg bg-amber-100 text-amber-800 flex items-center justify-center shrink-0">
                  <Upload className="w-3.5 h-3.5" />
                </div>
                <div>
                  <span className="font-semibold text-slate-900">Document library is empty</span>
                  <p className="text-slate-500 text-[11px] mt-0.5">
                    Upload PDF documents to enable grounded answers from your library.
                  </p>
                </div>
              </div>
              {onNavigateToDocs && (
                <button
                  onClick={onNavigateToDocs}
                  className="inline-flex items-center gap-1 text-xs font-semibold text-slate-900 hover:text-indigo-600 shrink-0"
                >
                  <span>Go to Library</span>
                  <ArrowUpRight className="w-3.5 h-3.5" />
                </button>
              )}
            </div>
          )}

          {/* Welcome / Empty Conversation View */}
          <div className="flex-1 flex flex-col justify-center my-auto py-8">
            <div className="text-center space-y-3 max-w-lg mx-auto">
              <div className="w-11 h-11 rounded-xl bg-slate-100 border border-slate-200 text-slate-700 flex items-center justify-center mx-auto shadow-sm">
                <BookOpen className="w-5 h-5" />
              </div>
              <h2 className="text-xl font-semibold text-slate-900 tracking-tight">
                Ask anything about your documents
              </h2>
              <p className="text-xs text-slate-500 leading-relaxed max-w-md mx-auto">
                Inquire across your uploaded document library. Answers are grounded in verified source passages with precise citations.
              </p>
            </div>
          </div>

          {/* Question Composer */}
          <div className="w-full pt-4 pb-2">
            <form
              onSubmit={(e) => e.preventDefault()}
              className="relative bg-white border border-slate-200 rounded-2xl shadow-xs focus-within:border-slate-400 focus-within:shadow-sm transition-all p-2.5"
            >
              <textarea
                rows={2}
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                  }
                }}
                placeholder={
                  isKbReady
                    ? "Ask a question about your documents..."
                    : "Upload documents to start asking questions..."
                }
                disabled={!isKbReady}
                className="w-full resize-none bg-transparent px-3 py-1.5 text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed"
              />
              <div className="flex items-center justify-between pt-2 px-2 border-t border-slate-100">
                <span className="text-[11px] text-slate-400 flex items-center gap-1">
                  <Layers className="w-3 h-3 text-slate-400" />
                  Grounded in verified document passages
                </span>
                <button
                  type="submit"
                  disabled={!isKbReady || !question.trim()}
                  className="w-8 h-8 rounded-xl bg-slate-900 text-white flex items-center justify-center hover:bg-slate-800 disabled:opacity-30 disabled:hover:bg-slate-900 transition-colors shadow-xs"
                >
                  <Send className="w-3.5 h-3.5" />
                </button>
              </div>
            </form>
          </div>
        </div>

        {/* Desktop Contextual Document Rail (Secondary Info Area, 1 Col on Desktop, Hidden on Mobile) */}
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
                    <span className={`text-[10px] font-medium capitalize ${doc.status === "indexed" ? "text-emerald-700" : "text-rose-700"}`}>
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


