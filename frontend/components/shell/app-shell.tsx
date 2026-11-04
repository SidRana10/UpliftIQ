"use client";

import { usePathname, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useState, useSyncExternalStore } from "react";
import { Sheet, SheetContent, SheetDescription, SheetTitle } from "@/components/ui/sheet";
import { TooltipProvider } from "@/components/ui/tooltip";
import { parseDataset, workspaceFor } from "@/lib/datasets";
import { NAV_COOKIE, type NavPref } from "@/lib/nav";
import { SidebarContent } from "./sidebar";
import { TopBar } from "./top-bar";

const XL = "(min-width: 1280px)";
function useIsXl(): boolean {
  return useSyncExternalStore(
    (cb) => {
      const m = window.matchMedia(XL);
      m.addEventListener("change", cb);
      return () => m.removeEventListener("change", cb);
    },
    () => window.matchMedia(XL).matches,
    () => true,
  );
}

export function AppShell({ initialNav, children }: { initialNav: NavPref; children: React.ReactNode }) {
  const [pref, setPref] = useState<NavPref>(initialNav);
  const isXl = useIsXl();
  // no stored preference: expanded on wide screens, icon rail on tablets
  const collapsed = pref ? pref === "collapsed" : !isXl;
  const [mobileOpen, setMobileOpen] = useState(false);

  const pathname = usePathname();
  const params = useSearchParams();
  const w = workspaceFor(pathname);
  const dataset = w?.datasets ? parseDataset(params.get("dataset") ?? undefined, w) : null;

  const toggle = useCallback(() => {
    const next: NavPref = collapsed ? "expanded" : "collapsed";
    setPref(next);
    document.cookie = `${NAV_COOKIE}=${next}; path=/; max-age=31536000; samesite=lax`;
  }, [collapsed]);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "b") {
        e.preventDefault();
        toggle();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggle]);

  return (
    <TooltipProvider delayDuration={200}>
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:fixed focus:left-3 focus:top-3 focus:z-50 focus:rounded-control focus:bg-surface focus:px-3 focus:py-2 focus:text-sm focus:shadow-popover"
      >
        Skip to content
      </a>
      <div className="min-h-dvh md:flex">
        <aside
          className={`on-dark sticky top-0 hidden h-dvh shrink-0 bg-sidebar transition-[width] duration-200 ease-out md:block ${
            collapsed ? "w-16" : "w-[248px]"
          }`}
        >
          <SidebarContent collapsed={collapsed} onToggle={toggle} />
        </aside>

        <div className="flex min-w-0 flex-1 flex-col">
          <TopBar onOpenNav={() => setMobileOpen(true)} dataset={dataset} />
          <main id="main" className="mx-auto w-full max-w-[1440px] flex-1 px-4 py-6 sm:px-6 xl:px-8 xl:py-8">
            {children}
          </main>
        </div>
      </div>

      <Sheet open={mobileOpen} onOpenChange={setMobileOpen}>
        <SheetContent aria-describedby="nav-desc">
          <SheetTitle className="sr-only">Navigation</SheetTitle>
          <SheetDescription id="nav-desc" className="sr-only">
            Switch between the five UpliftIQ workspaces.
          </SheetDescription>
          <SidebarContent collapsed={false} onNavigate={() => setMobileOpen(false)} />
        </SheetContent>
      </Sheet>
    </TooltipProvider>
  );
}
