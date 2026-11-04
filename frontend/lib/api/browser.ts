import { networkError, parseResponse, type Result } from "./errors";

/** Same-origin calls through the Next.js `/api/v1` proxy. Never throws. */
export async function browserGet<T>(path: string, signal?: AbortSignal): Promise<Result<T>> {
  try {
    const res = await fetch(`/api/v1${path}`, {
      headers: { Accept: "application/json" },
      cache: "no-store",
      signal,
    });
    return await parseResponse<T>(res);
  } catch (cause) {
    return { ok: false, error: networkError(cause) };
  }
}

export async function browserPost<T>(
  path: string,
  body: unknown,
  signal?: AbortSignal,
): Promise<Result<T>> {
  try {
    const res = await fetch(`/api/v1${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(body),
      cache: "no-store",
      signal,
    });
    return await parseResponse<T>(res);
  } catch (cause) {
    return { ok: false, error: networkError(cause) };
  }
}
