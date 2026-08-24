"use client";

import React, { useState, useCallback } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  ShieldCheck,
  MessageSquare,
  BarChart3,
  AlertTriangle,
  GitBranch,
  FlaskConical,
  Settings,
} from "lucide-react";
import { AppSidebar } from "@/components/layout/app-sidebar";
import { TopBar } from "@/components/shell/top-bar";
import { useAnalysisStore } from "@/store/analysis-store";

const MOBILE_NAV = [
  { href: "/overview", label: "Overview", icon: LayoutDashboard },
  { href: "/verify", label: "Verify", icon: ShieldCheck },
  { href: "/chat", label: "Chat", icon: MessageSquare },
  { href: "/errors", label: "Errors", icon: AlertTriangle },
  { href: "/traces", label: "Traces", icon: GitBranch },
];

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const sidebarOpen = useAnalysisStore((s) => s.sidebarOpen);
  const setSidebarOpen = useAnalysisStore((s) => s.setSidebarOpen);
  const pathname = usePathname();

  const handleCommandPaletteOpen = useCallback(() => {
    // Dispatch keyboard event to trigger cmdk
    const event = new KeyboardEvent("keydown", {
      key: "k",
      metaKey: true,
      bubbles: true,
    });
    document.dispatchEvent(event);
  }, []);

  return (
    <div className="app-shell">
      {/* Desktop Sidebar */}
      <AppSidebar isOpen={sidebarOpen} onToggle={() => setSidebarOpen(!sidebarOpen)} />

      {/* Main Content Area */}
      <div className="app-main">
        {/* Top Bar */}
        <TopBar onCommandPaletteOpen={handleCommandPaletteOpen} />

        {/* Page Content */}
        <main className="app-content" role="main">
          {children}
        </main>
      </div>

      {/* Mobile Bottom Navigation */}
      <nav
        className="fixed bottom-0 left-0 right-0 z-50 md:hidden bg-[var(--bg-surface)]/95 backdrop-blur-lg border-t border-[var(--border)] flex items-center justify-around h-14 px-1"
        role="navigation"
        aria-label="Mobile navigation"
      >
        {MOBILE_NAV.map((item) => {
          const isActive = pathname === item.href || (item.href !== "/" && pathname.startsWith(item.href));
          const Icon = item.icon;
          return (
            <Link key={item.href} href={item.href} className="flex-1">
              <div
                className={`flex flex-col items-center justify-center py-1 gap-0.5 cursor-pointer transition-colors ${
                  isActive
                    ? "text-[var(--primary)] font-semibold"
                    : "text-[var(--text-muted)] hover:text-[var(--text-secondary)]"
                }`}
              >
                <Icon className="w-[18px] h-[18px] shrink-0" />
                <span className="text-[9px] tracking-tight">{item.label}</span>
              </div>
            </Link>
          );
        })}
      </nav>
    </div>
  );
}
