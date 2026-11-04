export type ApiErrorKind =
  | "report_unavailable"
  | "model_unavailable"
  | "invalid_input"
  | "invalid_request"
  | "not_found"
  | "network"
  | "server";

export interface ApiError {
  kind: ApiErrorKind;
  status: number;
  /** human-readable message from the backend `detail` (or a network description) */
  message: string;
  /** structured 422 details, when present */
  fields?: { loc: string[]; msg: string; type: string }[];
}

export type Result<T> = { ok: true; data: T } | { ok: false; error: ApiError };

const KINDS: Record<string, ApiErrorKind> = {
  report_unavailable: "report_unavailable",
  model_unavailable: "model_unavailable",
  invalid_input: "invalid_input",
  invalid_request: "invalid_request",
};

/** Map a backend error response (`{detail, code}`) to a typed error. */
export function toApiError(status: number, body: unknown): ApiError {
  const b = (body ?? {}) as { detail?: unknown; code?: string };
  const kind: ApiErrorKind = (b.code && KINDS[b.code]) || (status === 404 ? "not_found" : "server");
  if (Array.isArray(b.detail)) {
    return {
      kind,
      status,
      message: "The request was not valid.",
      fields: b.detail as ApiError["fields"],
    };
  }
  const message =
    typeof b.detail === "string" ? b.detail : `The API returned an unexpected response (${status}).`;
  return { kind, status, message };
}

export function networkError(cause: unknown): ApiError {
  const detail = cause instanceof Error ? cause.message : "request failed";
  return { kind: "network", status: 0, message: `Could not reach the API (${detail}).` };
}

export async function parseResponse<T>(res: Response): Promise<Result<T>> {
  let body: unknown = null;
  try {
    body = await res.json();
  } catch {
    // non-JSON body: fall through with null
  }
  if (res.ok) return { ok: true, data: body as T };
  return { ok: false, error: toApiError(res.status, body) };
}
