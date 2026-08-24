"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import {
  LayoutDashboard,
  ShieldCheck,
  BarChart3,
  AlertTriangle,
  GitBranch,
  FlaskConical,
  Settings,
  MessageSquare,
  Activity,
  ChevronLeft,
  ChevronRight,
  Wrench,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useHealth } from "@/hooks/use-analysis";

const NAV_SECTIONS = [
  {
    title: "Platform",
    items: [
      { href: "/overview", label: "Overview", icon: LayoutDashboard, shortcut: "1" },
      { href: "/verify", label: "Verify", icon: ShieldCheck, shortcut: "2" },
      { href: "/chat", label: "Chat", icon: MessageSquare, shortcut: "3" },
      { href: "/evaluate", label: "Evaluate", icon: BarChart3, shortcut: "4" },
    ],
  },
  {
    title: "Observability",
    items: [
      { href: "/errors", label: "Error Feed", icon: AlertTriangle, shortcut: "5" },
      { href: "/traces", label: "Traces", icon: GitBranch, shortcut: "6" },
    ],
  },
  {
    title: "Research",
    items: [
      { href: "/scientific", label: "Scientific Lab", icon: FlaskConical, shortcut: "7" },
    ],
  },
];

const BOTTOM_ITEMS = [
  { href: "/settings", label: "Settings", icon: Settings },
  { href: "/admin", label: "Admin", icon: Wrench },
];

interface AppSidebarProps {
  isOpen: boolean;
  onToggle: () => void;
}

export function AppSidebar({ isOpen, onToggle }: AppSidebarProps) {
  const pathname = usePathname();
  const { data: health } = useHealth();
  const isHealthy = health?.status === "ok" || health?.status === "healthy";

  const isActive = (href: string) =>
    pathname === href || (href !== "/" && pathname.startsWith(href));

  return (
    <motion.aside
      initial={false}
      animate={{ width: isOpen ? "var(--sidebar-width)" : "var(--sidebar-collapsed)" }}
      transition={{ duration: 0.2, ease: [0.4, 0, 0.2, 1] }}
      className={cn(
        "relative hidden md:flex flex-col h-full",
        "border-r border-[var(--border)] bg-[var(--bg-surface)]",
        "overflow-hidden z-30 shrink-0"
      )}
      role="navigation"
      aria-label="Main navigation"
    >
      {/* ── Brand ────────────────────────────────────────────────────── */}
      <div className="flex items-center gap-2.5 px-3 h-[var(--topbar-height)] shrink-0 border-b border-[var(--border)]">
        <div className={cn(
          "flex items-center justify-center shrink-0",
          "w-8 h-8 rounded-[var(--radius)]",
          "bg-[var(--primary-soft)] border border-[var(--ai-border)]"
        )}>
          <ShieldCheck className="w-4 h-4 text-[var(--primary)]" />
        </div>
        <AnimatePresence>
          {isOpen && (
            <motion.div
              initial={{ opacity: 0, x: -6 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -6 }}
              transition={{ duration: 0.15 }}
              className="flex flex-col min-w-0"
            >
              <span className="text-sm font-semibold text-[var(--text-primary)] tracking-tight leading-none">
                HalluciSense
              </span>
              <span className="text-[10px] text-[var(--text-muted)] font-medium mt-0.5">
                v1.0.0
              </span>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* ── Navigation Sections ──────────────────────────────────────── */}
      <nav className="flex-1 overflow-y-auto py-3 px-2 space-y-4">
        {NAV_SECTIONS.map((section) => (
          <div key={section.title}>
            <AnimatePresence>
              {isOpen && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="px-2 mb-1"
                >
                  <span className="text-[10px] font-semibold uppercase tracking-widest text-[var(--text-dim)]">
                    {section.title}
                  </span>
                </motion.div>
              )}
            </AnimatePresence>
            <div className="space-y-0.5">
              {section.items.map((item) => {
                const active = isActive(item.href);
                const Icon = item.icon;
                return (
                  <Link key={item.href} href={item.href}>
                    <div
                      className={cn(
                        "group relative flex items-center gap-2.5 px-2.5 py-[7px] rounded-[var(--radius)] transition-all duration-150 cursor-pointer",
                        active
                          ? "bg-[var(--primary-soft)] text-[var(--primary)]"
                          : "text-[var(--text-secondary)] hover:bg-[var(--surface-hover)] hover:text-[var(--text-primary)]"
                      )}
                      role="menuitem"
                      aria-current={active ? "page" : undefined}
                    >
                      {/* Active indicator */}
                      {active && (
                        <motion.div
                          layoutId="sidebar-active-v2"
                          className="absolute left-0 top-1/2 -translate-y-1/2 w-[2.5px] h-4 rounded-full bg-[var(--primary)]"
                          transition={{ type: "spring", stiffness: 400, damping: 30 }}
                        />
                      )}

                      <Icon className={cn(
                        "w-[16px] h-[16px] shrink-0",
                        active ? "text-[var(--primary)]" : "text-[var(--text-muted)] group-hover:text-[var(--text-secondary)]"
                      )} />

                      <AnimatePresence>
                        {isOpen && (
                          <motion.div
                            initial={{ opacity: 0, x: -6 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: -6 }}
                            transition={{ duration: 0.12 }}
                            className="flex items-center justify-between flex-1 min-w-0"
                          >
                            <span className={cn(
                              "text-[13px] truncate",
                              active ? "font-semibold" : "font-normal"
                            )}>
                              {item.label}
                            </span>
                            {item.shortcut && (
                              <span className="text-[10px] font-mono text-[var(--text-dim)] opacity-0 group-hover:opacity-100 transition-opacity">
                                {item.shortcut}
                              </span>
                            )}
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>
                  </Link>
                );
              })}
            </div>
          </div>
        ))}
      </nav>

      {/* ── Bottom Section ───────────────────────────────────────────── */}
      <div className="border-t border-[var(--border)] px-2 py-2 space-y-0.5 shrink-0">
        {BOTTOM_ITEMS.map((item) => {
          const active = isActive(item.href);
          const Icon = item.icon;
          return (
            <Link key={item.href} href={item.href}>
              <div
                className={cn(
                  "flex items-center gap-2.5 px-2.5 py-[7px] rounded-[var(--radius)] transition-all duration-150 cursor-pointer",
                  active
                    ? "bg-[var(--primary-soft)] text-[var(--primary)]"
                    : "text-[var(--text-muted)] hover:bg-[var(--surface-hover)] hover:text-[var(--text-secondary)]"
                )}
              >
                <Icon className="w-[16px] h-[16px] shrink-0" />
                <AnimatePresence>
                  {isOpen && (
                    <motion.span
                      initial={{ opacity: 0, x: -6 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -6 }}
                      transition={{ duration: 0.12 }}
                      className={cn("text-[13px] truncate", active && "font-semibold")}
                    >
                      {item.label}
                    </motion.span>
                  )}
                </AnimatePresence>
              </div>
            </Link>
          );
        })}

        {/* System Status Footer */}
        <div className="flex items-center gap-2.5 px-2.5 py-2 mt-1">
          <div className="relative shrink-0">
            <Activity className="w-3.5 h-3.5 text-[var(--text-dim)]" />
            <span
              className={cn(
                "absolute -top-0.5 -right-0.5 w-[6px] h-[6px] rounded-full",
                isHealthy ? "bg-[var(--verified)]" : "bg-[var(--hallucination)]"
              )}
              aria-hidden="true"
            />
          </div>
          <AnimatePresence>
            {isOpen && (
              <motion.span
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="text-[11px] text-[var(--text-dim)]"
              >
                {isHealthy ? "All systems operational" : "Systems offline"}
              </motion.span>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* ── Toggle Button ────────────────────────────────────────────── */}
      <button
        onClick={onToggle}
        className={cn(
          "absolute top-[68px] -right-3 z-40 hidden md:flex",
          "items-center justify-center w-6 h-6 rounded-full",
          "bg-[var(--bg-raised)] border border-[var(--border-strong)]",
          "text-[var(--text-muted)] hover:text-[var(--text-primary)]",
          "transition-all duration-150 cursor-pointer shadow-[var(--shadow-sm)]"
        )}
        aria-label={isOpen ? "Collapse sidebar" : "Expand sidebar"}
      >
        {isOpen ? <ChevronLeft className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
      </button>
    </motion.aside>
  );
}
