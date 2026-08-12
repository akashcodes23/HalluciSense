"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { ShieldCheck, Zap, LayoutDashboard, GitBranch, BarChart3 } from "lucide-react";
import { AppSidebar } from "@/components/layout/app-sidebar";
import { useAnalysisStore } from "@/store/analysis-store";

const NAV_ITEMS = [
  { href: "/verify", label: "Verify", icon: ShieldCheck },
  { href: "/analyze", label: "Analyzer", icon: Zap },
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { href: "/traces", label: "Traces", icon: GitBranch },
  { href: "/metrics", label: "Metrics", icon: BarChart3 },
];

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const sidebarOpen = useAnalysisStore((s) => s.sidebarOpen);
  const setSidebarOpen = useAnalysisStore((s) => s.setSidebarOpen);
  const pathname = usePathname();

  return (
    <div className="min-h-screen bg-[#050816] text-slate-100 flex overflow-hidden font-sans relative">
      <AppSidebar isOpen={sidebarOpen} onToggle={() => setSidebarOpen(!sidebarOpen)} />
      <main className="min-w-0 flex-1 h-screen overflow-y-auto relative bg-[#050816] pb-20 md:pb-0">
        {children}
      </main>

      {/* Mobile Bottom Navigation */}
      <nav className="fixed bottom-0 left-0 right-0 z-50 md:hidden bg-[#060a14]/95 backdrop-blur-lg border-t border-white/[0.06] flex items-center justify-around h-16 px-2 shadow-2xl">
        {NAV_ITEMS.map((item) => {
          const isActive = pathname === item.href || (item.href !== "/" && pathname.startsWith(item.href));
          const Icon = item.icon;
          return (
            <Link key={item.href} href={item.href} className="flex-1">
              <div className={`flex flex-col items-center justify-center py-1 gap-1 cursor-pointer transition-colors ${isActive ? "text-blue-400 font-semibold" : "text-slate-400 hover:text-slate-200"}`}>
                <Icon className="w-5 h-5 shrink-0" />
                <span className="text-[10px] tracking-tight">{item.label}</span>
              </div>
            </Link>
          );
        })}
      </nav>
    </div>
  );
}
