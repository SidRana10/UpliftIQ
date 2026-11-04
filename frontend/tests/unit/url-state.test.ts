import { describe, expect, it } from "vitest";
import { parseList } from "@/lib/url-state";

const ALLOWED = ["a", "b", "c", "d"] as const;

describe("parseList", () => {
  it("keeps allowed values in order, de-duplicated", () => {
    expect(parseList("b,a,b", ALLOWED, 3, ["c"])).toEqual(["b", "a"]);
  });
  it("drops unknown values and caps the selection", () => {
    expect(parseList("a,zzz,b,c,d", ALLOWED, 3, ["c"])).toEqual(["a", "b", "c"]);
  });
  it("falls back when empty, missing or entirely invalid", () => {
    expect(parseList(null, ALLOWED, 3, ["c", "d"])).toEqual(["c", "d"]);
    expect(parseList("", ALLOWED, 3, ["c"])).toEqual(["c"]);
    expect(parseList("x,y", ALLOWED, 3, ["d"])).toEqual(["d"]);
  });
  it("returns a copy of the fallback, not the fallback itself", () => {
    const fb = ["c"] as const;
    expect(parseList(null, ALLOWED, 3, fb)).not.toBe(fb);
  });
});
