import { describe, expect, it } from "vitest";
import { networkError, parseResponse, toApiError } from "@/lib/api/errors";

describe("backend error mapping", () => {
  it("maps the backend error codes to typed kinds", () => {
    expect(toApiError(404, { detail: "report not available: x", code: "report_unavailable" })).toMatchObject({
      kind: "report_unavailable",
      status: 404,
      message: "report not available: x",
    });
    expect(toApiError(503, { detail: "model not loaded", code: "model_unavailable" }).kind).toBe("model_unavailable");
    expect(toApiError(422, { detail: "batch too large", code: "invalid_input" })).toMatchObject({
      kind: "invalid_input",
      message: "batch too large",
    });
  });

  it("keeps field-level details for 422 invalid_request", () => {
    const detail = [{ loc: ["body", "f0"], msg: "Field required", type: "missing" }];
    const e = toApiError(422, { detail, code: "invalid_request" });
    expect(e.kind).toBe("invalid_request");
    expect(e.fields).toEqual(detail);
  });

  it("distinguishes an unknown 404 from a server error", () => {
    expect(toApiError(404, { detail: "unknown figure" }).kind).toBe("not_found");
    expect(toApiError(500, null)).toMatchObject({ kind: "server", status: 500 });
  });

  it("describes network failures", () => {
    const e = networkError(new TypeError("fetch failed"));
    expect(e).toMatchObject({ kind: "network", status: 0 });
    expect(e.message).toContain("fetch failed");
  });
});

describe("parseResponse", () => {
  it("returns data for ok responses and typed errors otherwise", async () => {
    const ok = await parseResponse<{ a: number }>(new Response(JSON.stringify({ a: 1 }), { status: 200 }));
    expect(ok).toEqual({ ok: true, data: { a: 1 } });

    const bad = await parseResponse(
      new Response(JSON.stringify({ detail: "nope", code: "report_unavailable" }), { status: 404 }),
    );
    expect(bad.ok).toBe(false);
    if (!bad.ok) expect(bad.error.kind).toBe("report_unavailable");
  });

  it("survives a non-JSON error body", async () => {
    const r = await parseResponse(new Response("<html>bad gateway</html>", { status: 502 }));
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.error).toMatchObject({ kind: "server", status: 502 });
  });
});
