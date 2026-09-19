import React, { useEffect, useState } from "react";
import { Header, type ActiveTab } from "./components/layout/Header";
import { KnowledgeBaseScreen } from "./components/screens/KnowledgeBaseScreen";
import { ChatScreen } from "./components/screens/ChatScreen";
import { TraceScreen } from "./components/screens/TraceScreen";
import type { DocumentRecord, KnowledgeBaseStatus } from "./types/api";
import { getDocuments, getKnowledgeBaseStatus } from "./services/api";
import { Sparkles } from "lucide-react";

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<ActiveTab>("chat");
  const [kbStatus, setKbStatus] = useState<KnowledgeBaseStatus | null>(null);
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [isBackendConnected, setIsBackendConnected] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  const fetchBackendData = async () => {
    setIsLoading(true);
    try {
      const [statusRes, docsRes] = await Promise.all([
        getKnowledgeBaseStatus(),
        getDocuments(),
      ]);
      setKbStatus(statusRes);
      setDocuments(docsRes.documents);
      setIsBackendConnected(true);
    } catch (err) {
      console.warn("Backend connection failed:", err);
      setIsBackendConnected(false);
      setKbStatus(null);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchBackendData();
  }, []);

  return (
    <div className="min-h-screen bg-slate-50/60 text-slate-900 flex flex-col font-sans">
      {/* Subtle Application Shell Header */}
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        kbState={kbStatus?.state ?? "empty"}
        isBackendConnected={isBackendConnected}
      />

      {/* Main Content Viewport */}
      <main className="flex-1 max-w-6xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 flex flex-col">
        {isLoading && !kbStatus && isBackendConnected ? (
          <div className="py-20 text-center my-auto">
            <div className="inline-block h-6 w-6 animate-spin rounded-full border-2 border-indigo-600 border-t-transparent" />
            <p className="mt-3 text-xs font-medium text-slate-500">
              Connecting to Assistant...
            </p>
          </div>
        ) : (
          <>
            {activeTab === "chat" && (
              <ChatScreen status={kbStatus} onNavigateToDocs={() => setActiveTab("kb")} />
            )}

            {activeTab === "kb" && (
              <KnowledgeBaseScreen
                status={kbStatus}
                documents={documents}
                onRefresh={fetchBackendData}
              />
            )}

            {activeTab === "trace" && <TraceScreen />}
          </>
        )}
      </main>

      {/* Clean Footer */}
      <footer className="border-t border-slate-200/80 bg-white py-4 text-xs text-slate-500">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <Sparkles className="w-3.5 h-3.5 text-indigo-600" />
            <span className="font-medium text-slate-700">Mini AKA</span>
            <span>•</span>
            <span>Grounded AI Document Assistant</span>
          </div>
          <div className="text-[11px] text-slate-400">
            {isBackendConnected ? "Connected" : "Disconnected"}
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
