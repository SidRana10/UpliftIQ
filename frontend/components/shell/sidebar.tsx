"use client";

import {
  ChartSpline,
  FlaskConical,
  LayoutDashboard,
  PanelLeftClose,
  PanelLeftOpen,
  ServerCog,
  Target,
  type LucideIcon,
} from "lucide-react";
import Link from "next/link";
import { usePathname, useSearchParams } from "next/navigation";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { WORKSPACES, hrefWithDataset, isDataset, workspaceFor, type WorkspaceId } from "@/lib/datasets";
import { cn } from "@/lib/utils";
import { ApiStatus } from "./api-status";

const ICONS: Record<WorkspaceId, LucideIcon> = {
  overview: LayoutDashboard,
  experiments: FlaskConical,
  uplift: ChartSpline,
  decisions: Target,
  operations: ServerCog,
};

export function BrandMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" aria-hidden className={cn("size-8 shrink-0", className)}>
      <rect width="32" height="32" rx="8" fill="#117E74" />
      <path
        d="M8 21.5 13.5 15l4 3.5L24 10.5"
        fill="none"
        stroke="#fff"
        strokeWidth="2.4"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path d="M19 10.5h5v5" fill="none" stroke="#fff" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function SidebarContent({
  collapsed,
  onToggle,
  onNavigate,
}: {
  collapsed: boolean;
  /** omitted inside the mobile sheet (no collapse control there) */
  onToggle?: () => void;
  onNavigate?: () => void;
}) {
  const pathname = usePathname();
  const params = useSearchParams();
  const raw = params.get("dataset");
  const dataset = isDataset(raw) ? raw : null;
  const active = workspaceFor(pathname)?.id;

  return (
    <div className="flex h-full min-h-0 flex-col text-sidebar-fg">
      <div
        className={cn(
          "flex h-14 shrink-0 items-center gap-3 border-b border-sidebar-border",
          collapsed ? "justify-center px-0" : "px-4",
        )}
      >
        <BrandMark />
        {!collapsed ? (
          <div className="min-w-0 leading-tight">
            <div className="text-[15px] font-semibold tracking-[-0.01em] text-white">UpliftIQ</div>
            <div className="truncate text-[11px] text-sidebar-muted">Causal decision intelligence</div>
          </div>
        ) : null}
      </div>

      <nav aria-label="Workspaces" className="scrollbar-thin min-h-0 flex-1 overflow-y-auto px-2.5 py-4">
        {!collapsed ? (
          <div className="px-2.5 pb-2 text-[11px] font-medium uppercase tracking-[0.06em] text-sidebar-muted">
            Workspaces
          </div>
        ) : null}
        <ul className="space-y-0.5">
          {WORKSPACES.map((w) => {
            const Icon = ICONS[w.id];
            const isActive = active === w.id;
            const link = (
              <Link
                href={hrefWithDataset(w, dataset)}
                onClick={onNavigate}
                aria-current={isActive ? "page" : undefined}
                aria-label={collapsed ? `${w.number} ${w.label}` : undefined}
                className={cn(
                  "group relative flex h-10 items-center gap-3 rounded-control text-[13.5px] font-medium transition-colors duration-150",
                  collapsed ? "justify-center px-0" : "px-2.5",
                  isActive
                    ? "bg-sidebar-elevated text-white"
                    : "text-sidebar-fg hover:bg-sidebar-elevated/60 hover:text-white",
                )}
              >
                {isActive ? (
                  <span aria-hidden className="absolute inset-y-2 left-0 w-0.5 rounded-full bg-[#35c4b3]" />
                ) : null}
                <Icon className={cn("size-[18px] shrink-0", isActive ? "text-[#5fd6c8]" : "text-sidebar-muted group-hover:text-sidebar-fg")} />
                {!collapsed ? (
                  <>
                    <span className="flex-1 truncate">{w.label}</span>
                    <span className="font-mono text-[10.5px] text-sidebar-muted">{w.number}</span>
                  </>
                ) : null}
              </Link>
            );
            return (
              <li key={w.id}>
                {collapsed ? (
                  <Tooltip>
                    <TooltipTrigger asChild>{link}</TooltipTrigger>
                    <TooltipContent side="right">
                      {w.number} · {w.label}
                    </TooltipContent>
                  </Tooltip>
                ) : (
                  link
                )}
              </li>
            );
          })}
        </ul>
      </nav>

      <div className="shrink-0 space-y-1 border-t border-sidebar-border p-2.5">
        <ApiStatus collapsed={collapsed} />
        {onToggle ? (
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                type="button"
                onClick={onToggle}
                aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
                aria-keyshortcuts="Control+B Meta+B"
                className={cn(
                  "flex h-9 w-full items-center gap-3 rounded-control text-[13px] text-sidebar-muted transition-colors hover:bg-sidebar-elevated/60 hover:text-white",
                  collapsed ? "justify-center" : "px-2.5",
                )}
              >
                {collapsed ? <PanelLeftOpen className="size-[18px]" /> : <PanelLeftClose className="size-[18px]" />}
                {!collapsed ? (
                  <>
                    <span className="flex-1 text-left">Collapse</span>
                    <kbd className="rounded border border-sidebar-border px-1.5 font-mono text-[10px] text-sidebar-muted">⌘B</kbd>
                  </>
                ) : null}
              </button>
            </TooltipTrigger>
            <TooltipContent side="right">{collapsed ? "Expand sidebar" : "Collapse sidebar"} (⌘/Ctrl+B)</TooltipContent>
          </Tooltip>
        ) : null}
      </div>
    </div>
  );
}
