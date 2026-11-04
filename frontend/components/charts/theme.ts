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
