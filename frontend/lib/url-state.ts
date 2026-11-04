"use client";

import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useCallback } from "react";

/** One query-string value validated against an allow-list. View state lives in the URL. */
export function useQueryState<T extends string>(
  key: string,
  fallback: T,
  allowed: readonly T[],
): [T, (next: T) => void] {
  const params = useSearchParams();
  const pathname = usePathname();
  const router = useRouter();
  const raw = params.get(key);
  const value = allowed.includes(raw as T) ? (raw as T) : fallback;

  const set = useCallback(
    (next: T) => {
      const q = new URLSearchParams(params.toString());
      q.set(key, next);
      router.replace(`${pathname}?${q.toString()}`, { scroll: false });
    },
    [params, pathname, router, key],
  );

  return [value, set];
}
