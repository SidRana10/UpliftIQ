import { networkError, parseResponse, type Result } from "./errors";

const BASE = process.env.UPLIFTIQ_API_URL ?? "http://127.0.0.1:8000";
const REPORT_REVALIDATE_S = 3600;

/**
 * GET from a Server Component. Never throws: failures come back as a typed `Result` so a section
 * can render its own error state without taking the page down.
 *
 * Stored-report endpoints are immutable historical data (cached for an hour); live endpoints
 * (`/health`, `/models*`) pass `live: true` and are never cached.
 */
export async function apiGet<T>(path: string, opts: { live?: boolean } = {}): Promise<Result<T>> {
  try {
    const res = await fetch(`${BASE}/api/v1${path}`, {
      headers: { Accept: "application/json" },
      ...(opts.live ? { cache: "no-store" } : { next: { revalidate: REPORT_REVALIDATE_S } }),
    });
    return await parseResponse<T>(res);
  } catch (cause) {
    return { ok: false, error: networkError(cause) };
  }
}
