"use client";

import React from "react";
import { AppSidebar } from "@/components/layout/app-sidebar";
import { useAnalysisStore } from "@/store/analysis-store";

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const sidebarOpen = useAnalysisStore((s) => s.sidebarOpen);
  const setSidebarOpen = useAnalysisStore((s) => s.setSidebarOpen);

  return (
    <div className="min-h-screen bg-[#050816] text-slate-100 flex overflow-hidden font-sans">
      <AppSidebar isOpen={sidebarOpen} onToggle={() => setSidebarOpen(!sidebarOpen)} />
      <main className="min-w-0 flex-1 h-screen overflow-y-auto relative bg-[#050816]">
        {children}
      </main>
    </div>
  );
}
