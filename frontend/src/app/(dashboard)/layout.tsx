"use client";

import React, { useState } from "react";
import { AppSidebar } from "@/components/layout/app-sidebar";
import { useAnalysisStore } from "@/store/analysis-store";

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const sidebarOpen = useAnalysisStore((s) => s.sidebarOpen);
  const setSidebarOpen = useAnalysisStore((s) => s.setSidebarOpen);

  return (
    <div className="app-shell">
      <AppSidebar isOpen={sidebarOpen} onToggle={() => setSidebarOpen(!sidebarOpen)} />

      <main className="flex-1 flex flex-col h-dvh overflow-hidden min-w-0 relative">
        {children}
      </main>
    </div>
  );
}
