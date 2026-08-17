"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import {
  MessageSquare,
  ShieldCheck,
  Zap,
  GitBranch,
  BarChart3,
  Settings,
  ChevronLeft,
  ChevronRight,
  Activity,
  LayoutDashboard,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useHealth } from "@/hooks/use-analysis";

const NAV_ITEMS = [
  { href: "/chat", label: "Chat", icon: MessageSquare, description: "Answer + Verification + Correction" },
  { href: "/verify", label: "Verify", icon: ShieldCheck, description: "Verify LLM responses" },
  { href: "/analyze", label: "Analyzer", icon: Zap, description: "Detailed analysis" },
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard, description: "Overview" },
  { href: "/traces", label: "Traces", icon: GitBranch, description: "Pipeline traces" },
  { href: "/metrics", label: "Metrics", icon: BarChart3, description: "System metrics" },
  { href: "/settings", label: "Settings", icon: Settings, description: "Configuration" },
];

interface AppSidebarProps {
  isOpen: boolean;
  onToggle: () => void;
}

export function AppSidebar({ isOpen, onToggle }: AppSidebarProps) {
  const pathname = usePathname();
  const { data: health } = useHealth();
  const isHealthy = health?.status === "ok" || health?.status === "healthy";

  return (
    <motion.aside
      initial={false}
      animate={{ width: isOpen ? 240 : 72 }}
      transition={{ duration: 0.2, ease: [0.4, 0, 0.2, 1] }}
      className="relative hidden md:flex flex-col h-full border-r border-white/[0.06] bg-[#060a14] overflow-hidden z-30 shrink-0"
    >
      {/* ── Brand ──────────────────────────────────────────────────────── */}
      <div className="flex items-center gap-3 px-4 h-16 shrink-0 border-b border-white/[0.06]">
        <div className="flex items-center justify-center w-9 h-9 rounded-xl bg-accent-primary/15 border border-accent-primary/30 text-accent-primary">
          <ShieldCheck className="w-5 h-5" />
        </div>
        <AnimatePresence>
          {isOpen && (
            <motion.div
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -8 }}
              transition={{ duration: 0.15 }}
              className="flex flex-col min-w-0"
            >
              <span className="text-sm font-bold text-white tracking-tight">HalluciSense</span>
              <span className="text-[10px] text-slate-500 font-medium">Research v1.0 Production</span>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* ── Navigation ─────────────────────────────────────────────────── */}
      <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
        {NAV_ITEMS.map((item) => {
          const isActive = pathname === item.href || (item.href !== "/" && pathname.startsWith(item.href));
          const Icon = item.icon;

          return (
            <Link key={item.href} href={item.href}>
              <div
                className={cn(
                  "group relative flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all duration-150 cursor-pointer",
                  isActive
                    ? "bg-accent-primary/10 text-accent-primary font-semibold"
                    : "text-slate-400 hover:bg-white/[0.04] hover:text-slate-200 font-normal"
                )}
              >
                {isActive && (
                  <motion.div
                    layoutId="sidebar-active"
                    className="absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-5 rounded-full bg-accent-primary"
                    transition={{ type: "spring", stiffness: 400, damping: 30 }}
                  />
                )}

                <Icon className={cn("w-[18px] h-[18px] shrink-0", isActive ? "text-accent-primary" : "text-slate-400")} />

                <AnimatePresence>
                  {isOpen && (
                    <motion.span
                      initial={{ opacity: 0, x: -8 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -8 }}
                      transition={{ duration: 0.15 }}
                      className="text-sm truncate"
                    >
                      {item.label}
                    </motion.span>
                  )}
                </AnimatePresence>
              </div>
            </Link>
          );
        })}
      </nav>

      {/* ── Status Footer ──────────────────────────────────────────────── */}
      <div className="px-3 py-3 border-t border-white/[0.06] shrink-0">
        <div className="flex items-center gap-3 px-3 py-2">
          <div className="relative">
            <Activity className="w-4 h-4 text-slate-500" />
            <div
              className={cn(
                "absolute -top-0.5 -right-0.5 w-2 h-2 rounded-full",
                isHealthy ? "bg-emerald-500" : "bg-red-500"
              )}
            />
          </div>
          <AnimatePresence>
            {isOpen && (
              <motion.span
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="text-xs text-slate-500"
              >
                {isHealthy ? "Backend Active" : "Offline"}
              </motion.span>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* ── Toggle Button ──────────────────────────────────────────────── */}
      <button
        onClick={onToggle}
        className="absolute top-20 -right-3 z-40 hidden md:flex items-center justify-center w-6 h-6 rounded-full bg-[#111827] border border-white/[0.1] text-slate-400 hover:text-white transition-all cursor-pointer shadow-lg"
        aria-label={isOpen ? "Collapse sidebar" : "Expand sidebar"}
      >
        {isOpen ? <ChevronLeft className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
      </button>
    </motion.aside>
  );
}
