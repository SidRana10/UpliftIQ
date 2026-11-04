/* eslint-disable @typescript-eslint/no-explicit-any -- test-only helpers over untyped report JSON */
/**
 * Optional integration check: with the backend running, the payloads the pages receive must equal
 * the committed reports. Skipped unless UPLIFTIQ_VERIFY_API is set, e.g.
 *   UPLIFTIQ_VERIFY_API=http://127.0.0.1:8000 npm test -- live-api-parity
 */
import { describe, expect, it } from "vitest";
import * as R from "./report-data";

const API = process.env.UPLIFTIQ_VERIFY_API;

async function live(path: string): Promise<any> {
  const res = await fetch(`${API}/api/v1${path}`);
  expect(res.ok, path).toBe(true);
  return (await res.json()).data;
}

/** deep compare; floats may differ by the API's 10-decimal JSON rounding */
function same(a: unknown, b: unknown, path = "$"): void {
  if (typeof a === "number" && typeof b === "number") {
    expect(Math.abs(a - b), `${path}: ${a} vs ${b}`).toBeLessThanOrEqual(1e-9 + 1e-9 * Math.abs(b));
  } else if (Array.isArray(a) && Array.isArray(b)) {
    expect(a.length, `${path} length`).toBe(b.length);
    a.forEach((x, i) => same(x, b[i], `${path}[${i}]`));
  } else if (a && b && typeof a === "object" && typeof b === "object") {
    for (const k of Object.keys(b as object)) same((a as any)[k], (b as any)[k], `${path}.${k}`);
  } else {
    expect(a, path).toEqual(b);
  }
}

describe.skipIf(!API || !R.reportsAvailable)("live API payloads equal the committed reports", () => {
  it("Criteo benchmark", async () => same(await live("/criteo/benchmark"), R.criteoBenchmark()));
  it("Criteo held-out", async () => same(await live("/criteo/heldout"), R.criteoHeldout()));
  it("Criteo IPW vs AIPW", async () => same(await live("/criteo/ipw-aipw"), R.criteoIpwAipw()));
  it("Criteo diagnostics", async () => same(await live("/criteo/diagnostics"), R.criteoDiagnostics()));
  it("Hillstrom effects", async () => same(await live("/hillstrom/effects"), R.hillstromEffects()));
  it("Hillstrom groups", async () => same(await live("/hillstrom/groups"), R.hillstromGroups()));
  it("Hillstrom balance", async () => same(await live("/hillstrom/balance"), R.hillstromBalance()));
  it("Hillstrom models", async () => same(await live("/hillstrom/models"), R.hillstromModels()));
});
