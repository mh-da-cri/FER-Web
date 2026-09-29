"use client";

import { useState } from "react";
import CameraStream from "@/components/CameraStream";
import ResultsPanel from "@/components/ResultsPanel";

export default function HomePage() {
  const [analysisResult, setAnalysisResult] = useState(null);
  const [cameraStatus, setCameraStatus] = useState("off");

  return (
    <div className="app-container">
      {/* ─── Header ─── */}
      <header className="app-header">
        <div className="app-header__badge">
          AI-Powered • Real-time Analysis
        </div>
        <h1 className="app-header__title">AI Camera</h1>
        <p className="app-header__subtitle">
          Phân tích biểu cảm khuôn mặt theo thời gian thực
        </p>
      </header>

      {/* ─── Main Content ─── */}
      <main className="main-grid">
        {/* Camera Feed */}
        <CameraStream
          onResult={setAnalysisResult}
          onStatusChange={setCameraStatus}
        />

        {/* Results Sidebar */}
        <ResultsPanel result={analysisResult} />
      </main>
    </div>
  );
}
