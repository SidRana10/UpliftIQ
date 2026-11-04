"use client";

import { useSyncExternalStore } from "react";
import { COLORS } from "@/lib/colors";

/** Shared chart styling (plan §A6): horizontal gridlines only, muted 12px axis text. */
export const CHART = {
  margin: { top: 12, right: 24, bottom: 8, left: 4 },
  axisTick: { fontSize: 12, fill: COLORS.axis },
  axisLine: { stroke: COLORS.zero },
  grid: { stroke: COLORS.grid, vertical: false as const },
  height: { primary: 320, secondary: 240, mobile: 220 },
};

const RM = "(prefers-reduced-motion: reduce)";

/** Charts animate on mount only, and not at all when the user prefers reduced motion. */
export function useReducedMotion(): boolean {
  return useSyncExternalStore(
    (cb) => {
      const m = window.matchMedia(RM);
      m.addEventListener("change", cb);
      return () => m.removeEventListener("change", cb);
    },
    () => window.matchMedia(RM).matches,
    () => true,
  );
}

const NARROW = "(max-width: 639px)";
export function useIsNarrow(): boolean {
  return useSyncExternalStore(
    (cb) => {
      const m = window.matchMedia(NARROW);
      m.addEventListener("change", cb);
      return () => m.removeEventListener("change", cb);
    },
    () => window.matchMedia(NARROW).matches,
    () => false,
  );
}

/** Round an axis outward to a 1/2/5 step so ticks land on readable numbers. */
export function niceAxis(lo: number, hi: number, target = 5): { domain: [number, number]; ticks: number[] } {
  const span = hi - lo || 1;
  const raw = span / target;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const start = Math.floor(lo / step) * step;
  const end = Math.ceil(hi / step) * step;
  const ticks: number[] = [];
  for (let v = start; v <= end + step / 2; v += step) ticks.push(Number(v.toFixed(10)));
  return { domain: [start, end], ticks };
}
