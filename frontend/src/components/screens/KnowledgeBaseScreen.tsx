import React, { useEffect, useState } from "react";
import { Upload, FileText, AlertCircle, Library, Plus } from "lucide-react";
import type { DocumentRecord, KnowledgeBaseStatus } from "../../types/api";
import type { TraceEvent } from "../../types/events";
import { uploadDocument } from "../../services/api";

interface KnowledgeBaseScreenProps {
  status: KnowledgeBaseStatus | null;
  documents: DocumentRecord[];
  onRefresh: () => void;
}

export const KnowledgeBaseScreen: React.FC<KnowledgeBaseScreenProps> = ({
  documents,
  onRefresh,
}) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadEvents, setUploadEvents] = useState<TraceEvent[]>([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [showUploadModal, setShowUploadModal] = useState(false);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && showUploadModal && !isUploading) {
        setShowUploadModal(false);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [showUploadModal, isUploading]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      if (!file.name.toLowerCase().endsWith(".pdf")) {
        setErrorMessage("Only PDF documents (.pdf) are supported.");
        setSelectedFile(null);
        return;
      }
      setErrorMessage(null);
      setSelectedFile(file);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    setIsUploading(true);
    setUploadEvents([]);
    setErrorMessage(null);

    await uploadDocument(
      selectedFile,
      (evt) => {
        setUploadEvents((prev) => [...prev, evt]);
        if (evt.type === "COMPLETE" || evt.type === "DOCUMENT_DUPLICATE") {
          setIsUploading(false);
          setSelectedFile(null);
          setShowUploadModal(false);
          onRefresh();
        }
      },
      (err) => {
        setIsUploading(false);
        setErrorMessage(err.message);
      }
    );
  };

  return (
    <div className="space-y-6 max-w-6xl mx-auto w-full">
      {/* Document Library Header & Action Bar */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-4 border-b border-slate-200">
        <div>
          <h2 className="text-lg font-semibold text-slate-900 tracking-tight">
            Document Library
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Manage document sources used for grounded retrieval and citations.
          </p>
        </div>

        <button
          onClick={() => setShowUploadModal(true)}
          className="inline-flex items-center gap-2 bg-slate-900 hover:bg-slate-800 text-white text-xs font-medium px-4 py-2.5 rounded-xl shadow-xs transition-colors focus-visible:ring-2 focus-visible:ring-slate-400 focus-visible:outline-none"
        >
          <Plus className="w-4 h-4" />
          <span>Upload Document</span>
        </button>
      </div>

      {/* Upload Modal / Area */}
      {showUploadModal && (
        <div
          role="dialog"
          aria-modal="true"
          aria-labelledby="upload-modal-title"
          className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm space-y-4"
        >
          <div className="flex items-center justify-between border-b border-slate-100 pb-3">
            <h3 id="upload-modal-title" className="text-sm font-semibold text-slate-900">
              Upload PDF Document
            </h3>
            <button
              onClick={() => setShowUploadModal(false)}
              className="text-xs text-slate-400 hover:text-slate-600 font-medium focus-visible:ring-2 focus-visible:ring-slate-400 focus-visible:outline-none rounded"
            >
              Cancel
            </button>
          </div>

          <div className="border border-dashed border-slate-300 hover:border-slate-400 bg-slate-50/50 rounded-xl p-8 text-center transition-colors">
            <Upload className="w-7 h-7 text-slate-400 mx-auto mb-2" />
            <p className="text-xs font-medium text-slate-700">
              Select or drop a PDF document to index
            </p>
            <p className="text-[11px] text-slate-400 mt-1">
              Supports vector search & text extraction (Max 15 MB)
            </p>
            <input
              type="file"
              accept=".pdf"
              onChange={handleFileChange}
              className="hidden"
              id="pdf-library-upload-input"
            />
            <label
              htmlFor="pdf-library-upload-input"
              className="mt-4 inline-block px-3.5 py-1.5 bg-white border border-slate-300 rounded-lg text-xs font-medium text-slate-700 shadow-2xs cursor-pointer hover:bg-slate-50"
            >
              Choose File
            </label>
          </div>

          {selectedFile && (
            <div className="bg-slate-50 border border-slate-200 rounded-xl p-3 flex items-center justify-between text-xs">
              <div className="flex items-center gap-2 truncate">
                <FileText className="w-4 h-4 text-slate-700 shrink-0" />
                <span className="truncate font-medium text-slate-800">
                  {selectedFile.name}
                </span>
              </div>
              <span className="text-slate-500 text-[11px] shrink-0 font-mono">
                {(selectedFile.size / 1024).toFixed(1)} KB
              </span>
            </div>
          )}

          {errorMessage && (
            <div className="bg-rose-50 border border-rose-200 rounded-xl p-3 text-xs text-rose-700 flex items-center gap-2">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span>{errorMessage}</span>
            </div>
          )}

          {/* Ingestion Stream Progress */}
          {uploadEvents.length > 0 && (
            <div className="space-y-1.5 pt-2">
              {uploadEvents.map((evt) => (
                <div
                  key={evt.seq}
                  className="flex items-center justify-between text-xs text-slate-600 bg-slate-50 border border-slate-100 px-3 py-1.5 rounded-lg"
                >
                  <span className="font-medium">{evt.label}</span>
                  <span className="text-[10px] text-slate-400 font-mono">{evt.t_ms}ms</span>
                </div>
              ))}
            </div>
          )}

          <div className="flex justify-end gap-2 pt-2 border-t border-slate-100">
            <button
              onClick={() => setShowUploadModal(false)}
              className="px-3.5 py-2 border border-slate-300 rounded-xl text-xs font-medium text-slate-700 hover:bg-slate-50"
            >
              Cancel
            </button>
            <button
              disabled={!selectedFile || isUploading}
              onClick={handleUpload}
              className="px-4 py-2 bg-slate-900 hover:bg-slate-800 disabled:opacity-50 text-white rounded-xl text-xs font-medium shadow-xs transition-colors"
            >
              {isUploading ? "Indexing..." : "Upload & Process"}
            </button>
          </div>
        </div>
      )}

      {/* Document Inventory Table */}
      <div className="bg-white border border-slate-200 rounded-2xl overflow-hidden shadow-xs">
        {documents.length === 0 ? (
          <div className="py-16 text-center text-slate-400 px-4">
            <div className="w-11 h-11 rounded-xl bg-slate-100 text-slate-400 flex items-center justify-center mx-auto mb-3">
              <Library className="w-5 h-5" />
            </div>
            <h3 className="text-sm font-semibold text-slate-800">
              No documents in library
            </h3>
            <p className="text-xs text-slate-500 mt-1 max-w-sm mx-auto">
              Add PDF files to your library to enable grounded answers and source citations during document inquiries.
            </p>
            <button
              onClick={() => setShowUploadModal(true)}
              className="mt-4 inline-flex items-center gap-1.5 text-xs font-semibold text-slate-900 hover:text-indigo-600"
            >
              <Plus className="w-4 h-4" />
              <span>Add your first document</span>
            </button>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-slate-200 bg-slate-50/70 text-[11px] font-semibold text-slate-500 uppercase tracking-wider">
                  <th className="py-3 px-4">Document</th>
                  <th className="py-3 px-4">Pages</th>
                  <th className="py-3 px-4 font-mono">Sections</th>
                  <th className="py-3 px-4">Status</th>
                  <th className="py-3 px-4">Indexed Date</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {documents.map((doc) => (
                  <tr key={doc.doc_id} className="hover:bg-slate-50/60 transition-colors">
                    <td className="py-3.5 px-4 font-medium text-slate-900">
                      <div className="flex items-center gap-2.5">
                        <div className="w-7 h-7 rounded-lg bg-slate-100 text-slate-700 flex items-center justify-center shrink-0">
                          <FileText className="w-3.5 h-3.5" />
                        </div>
                        <span className="truncate max-w-xs">{doc.filename}</span>
                      </div>
                    </td>
                    <td className="py-3.5 px-4 text-slate-600">{doc.pages}</td>
                    <td className="py-3.5 px-4 text-slate-600 font-mono">{doc.chunks}</td>
                    <td className="py-3.5 px-4">
                      <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-medium bg-emerald-50 text-emerald-700 border border-emerald-200">
                        {doc.status}
                      </span>
                    </td>
                    <td className="py-3.5 px-4 text-slate-400 text-[11px] font-mono">
                      {new Date(doc.indexed_at).toLocaleDateString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

