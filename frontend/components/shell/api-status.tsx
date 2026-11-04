"use client";

import { useCallback, useEffect, useState } from "react";
import { browserGet } from "@/lib/api/browser";
import type { Health } from "@/lib/api/types";
import { cn } from "@/lib/utils";

type State =
  | { kind: "checking" }
  | { kind: "online"; health: Health }
  | { kind: "degraded"; health: Health }
  | { kind: "offline" };

const POLL_MS = 60_000;

/** Live API/model status from `GET /health` (never cached; refreshed every minute and on focus). */
export function useApiStatus(): State {
  const [state, setState] = useState<State>({ kind: "checking" });
  const check = useCallback(async (signal?: AbortSignal) => {
    const r = await browserGet<Health>("/health", signal);
    if (signal?.aborted) return;
    if (!r.ok) setState({ kind: "offline" });
    else setState({ kind: r.data.status === "ok" ? "online" : "degraded", health: r.data });
  }, []);

  useEffect(() => {
    const ctl = new AbortController();
    // checking on mount is intentional: the state machine starts at "checking"
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void check(ctl.signal);
    const id = setInterval(() => void check(ctl.signal), POLL_MS);
    const onFocus = () => void check(ctl.signal);
    window.addEventListener("focus", onFocus);
    return () => {
      ctl.abort();
      clearInterval(id);
      window.removeEventListener("focus", onFocus);
    };
  }, [check]);

  return state;
}

export function ApiStatus({ collapsed }: { collapsed: boolean }) {
  const state = useApiStatus();
  const { dot, label, detail } = describe(state);
  return (
    <div
      className="flex items-center gap-2.5 rounded-control px-2.5 py-2"
      role="status"
      aria-live="polite"
      title={`${label}${detail ? ` · ${detail}` : ""}`}
    >
      <span className="relative flex size-2 shrink-0">
        <span className={cn("size-2 rounded-full", dot)} />
      </span>
      {collapsed ? <span className="sr-only">{label}</span> : null}
      {!collapsed ? (
        <span className="min-w-0 leading-4">
          <span className="block truncate text-xs font-medium text-white">{label}</span>
          {detail ? <span className="block truncate text-[11px] text-sidebar-muted">{detail}</span> : null}
        </span>
      ) : null}
    </div>
  );
}

function describe(s: State): { dot: string; label: string; detail?: string } {
  switch (s.kind) {
    case "checking":
      return { dot: "bg-sidebar-muted animate-pulse motion-reduce:animate-none", label: "Checking API" };
    case "online":
      return {
        dot: "bg-[#35c4b3]",
        label: "API online",
        detail: s.health.model_version ? `Criteo model ${s.health.model_version}` : undefined,
      };
    case "degraded":
      return {
        dot: "bg-amber",
        label: "API degraded",
        detail: s.health.criteo_model_loaded ? "Some reports missing" : "Scoring model not loaded",
      };
    case "offline":
      return { dot: "bg-[#e5736d]", label: "API offline", detail: "Backend not reachable" };
  }
}
