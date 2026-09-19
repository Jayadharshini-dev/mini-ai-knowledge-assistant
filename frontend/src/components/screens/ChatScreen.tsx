import React, { useState } from "react";
import { Send, ArrowUpRight, Sparkles, Upload } from "lucide-react";
import type { KnowledgeBaseStatus } from "../../types/api";

interface ChatScreenProps {
  status: KnowledgeBaseStatus | null;
  onNavigateToDocs?: () => void;
}

export const ChatScreen: React.FC<ChatScreenProps> = ({
  status,
  onNavigateToDocs,
}) => {
  const [question, setQuestion] = useState("");
  const isKbReady = status?.state === "ready";

  return (
    <div className="flex-1 flex flex-col justify-between max-w-3xl mx-auto w-full min-h-[calc(100vh-10rem)]">
      {/* Empty KB Alert Notice */}
      {!isKbReady && (
        <div className="mb-6 bg-slate-100/80 border border-slate-200/80 rounded-xl p-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 text-xs text-slate-700">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg bg-amber-100 text-amber-700 flex items-center justify-center shrink-0">
              <Upload className="w-3.5 h-3.5" />
            </div>
            <div>
              <span className="font-semibold text-slate-900">Document library is empty</span>
              <p className="text-slate-500 text-[11px]">
                Upload PDF files to enable grounded answers from your documents.
              </p>
            </div>
          </div>
          {onNavigateToDocs && (
            <button
              onClick={onNavigateToDocs}
              className="inline-flex items-center gap-1 text-xs font-semibold text-indigo-600 hover:text-indigo-700 shrink-0"
            >
              <span>Add documents</span>
              <ArrowUpRight className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      )}

      {/* Main Conversation Container / Welcome Hero */}
      <div className="flex-1 flex flex-col justify-center my-auto py-8">
        <div className="text-center space-y-3 max-w-lg mx-auto">
          <div className="w-12 h-12 rounded-2xl bg-indigo-50 border border-indigo-100 text-indigo-600 flex items-center justify-center mx-auto shadow-sm">
            <Sparkles className="w-6 h-6" />
          </div>
          <h2 className="text-xl font-semibold text-slate-900 tracking-tight">
            Ask anything about your documents
          </h2>
          <p className="text-xs text-slate-500 leading-relaxed">
            Get instant, grounded answers with citations referenced directly from your uploaded document library.
          </p>
        </div>
      </div>

      {/* Centered Question Composer Input Box */}
      <div className="w-full pt-4 pb-2">
        <form
          onSubmit={(e) => e.preventDefault()}
          className="relative bg-white border border-slate-200/90 rounded-2xl shadow-sm focus-within:shadow-md focus-within:border-indigo-500 transition-all p-2"
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
          <div className="flex items-center justify-between pt-1 px-2 border-t border-slate-100">
            <span className="text-[11px] text-slate-400">
              Answers are backed by verified source passages
            </span>
            <button
              type="submit"
              disabled={!isKbReady || !question.trim()}
              className="w-8 h-8 rounded-xl bg-indigo-600 text-white flex items-center justify-center hover:bg-indigo-700 disabled:opacity-40 disabled:hover:bg-indigo-600 transition-colors shadow-sm"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
