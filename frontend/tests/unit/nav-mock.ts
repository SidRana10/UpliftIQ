/** TEST-ONLY: a reactive stand-in for next/navigation's useSearchParams over history.replaceState. */
import { useMemo, useSyncExternalStore } from "react";

const listeners = new Set<() => void>();
const original = window.history.replaceState.bind(window.history);
window.history.replaceState = (...args: Parameters<History["replaceState"]>) => {
  original(...args);
  listeners.forEach((l) => l());
};

export function useSearchParamsMock(): URLSearchParams {
  const search = useSyncExternalStore(
    (cb) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    () => window.location.search,
    () => "",
  );
  return useMemo(() => new URLSearchParams(search), [search]);
}

export const resetUrl = () => window.history.replaceState(null, "", "/");
