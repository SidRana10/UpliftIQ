"use client";

import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useCallback, useMemo } from "react";

/** One query-string value validated against an allow-list. View state lives in the URL. */
export function useQueryState<T extends string>(
  key: string,
  fallback: T,
  allowed: readonly T[],
  /** drop every other parameter when this one changes (e.g. view selectors of another dataset) */
  resetOthers = false,
): [T, (next: T) => void] {
  const params = useSearchParams();
  const pathname = usePathname();
  const router = useRouter();
  const raw = params.get(key);
  const value = allowed.includes(raw as T) ? (raw as T) : fallback;

  const set = useCallback(
    (next: T) => {
      const q = new URLSearchParams(resetOthers ? "" : params.toString());
      q.set(key, next);
      router.replace(`${pathname}?${q.toString()}`, { scroll: false });
    },
    [params, pathname, router, key, resetOthers],
  );

  return [value, set];
}

// ------------------------------------------------------------------ in-page view state

/** `a,b,c` -> allowed values only, de-duplicated, capped; empty/invalid input yields `fallback`. */
export function parseList<T extends string>(
  raw: string | null | undefined,
  allowed: readonly T[],
  max: number,
  fallback: readonly T[],
): T[] {
  const picked: T[] = [];
  for (const part of (raw ?? "").split(",")) {
    const v = part.trim() as T;
    if (allowed.includes(v) && !picked.includes(v)) picked.push(v);
    if (picked.length === max) break;
  }
  return picked.length ? picked : [...fallback];
}

function writeParam(key: string, value: string | null) {
  const q = new URLSearchParams(window.location.search);
  if (value === null) q.delete(key);
  else q.set(key, value);
  const qs = q.toString();
  // history.replaceState is integrated with Next's router: useSearchParams updates immediately and
  // no server round-trip is needed for a view that is already fully loaded.
  window.history.replaceState(null, "", `${window.location.pathname}${qs ? `?${qs}` : ""}`);
}

/** Like `useQueryState` but shallow: changes the URL without re-rendering the server page. */
export function useViewState<T extends string>(
  key: string,
  fallback: T,
  allowed: readonly T[],
): [T, (next: T) => void] {
  const params = useSearchParams();
  const raw = params.get(key);
  const value = allowed.includes(raw as T) ? (raw as T) : fallback;
  const set = useCallback((next: T) => writeParam(key, next), [key]);
  return [value, set];
}

/** A comma-separated multi-select (e.g. compared models) kept in one query parameter. */
export function useViewList<T extends string>(
  key: string,
  fallback: readonly T[],
  allowed: readonly T[],
  max: number,
): [T[], (next: T[]) => void] {
  const params = useSearchParams();
  const raw = params.get(key);
  const value = useMemo(() => parseList(raw, allowed, max, fallback), [raw, allowed, max, fallback]);
  const set = useCallback((next: T[]) => writeParam(key, next.join(",")), [key]);
  return [value, set];
}
