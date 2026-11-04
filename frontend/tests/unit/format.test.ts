import { describe, expect, it } from "vitest";
import {
  budget,
  compact,
  count,
  criteo,
  gigabytes,
  hillstrom,
  interval,
  ms,
  pValue,
  seconds,
  shortHash,
} from "@/lib/format";

describe("Hillstrom formatters (business units)", () => {
  it("formats rates as percent and differences as signed percentage points", () => {
    expect(hillstrom.rate(0.1819)).toBe("18.2%");
    expect(hillstrom.rate(0.0091, 2)).toBe("0.91%");
    expect(hillstrom.rateDelta(0.0294)).toBe("+2.94 pp");
    expect(hillstrom.rateDelta(-0.0113)).toBe("−1.13 pp");
  });

  it("does not print a sign for values that round to zero", () => {
    expect(hillstrom.rateDelta(0.00001)).toBe("0.00 pp");
    expect(hillstrom.rateDelta(-0.00001)).toBe("0.00 pp");
  });

  it("formats spend in dollars with signed deltas", () => {
    expect(hillstrom.spend(1.1556)).toBe("$1.16");
    expect(hillstrom.spendDelta(-0.1091)).toBe("−$0.11");
    expect(hillstrom.spendDelta(0.77)).toBe("+$0.77");
  });
});

describe("Criteo formatters (percentage points, never currency)", () => {
  it("converts stored fractions to percentage points", () => {
    expect(criteo.pp(0.006741)).toBe("0.674 pp");
    expect(criteo.ppValue(0.006741)).toBe("0.674");
    expect(criteo.ppDelta(0.005253)).toBe("+0.525 pp");
    expect(criteo.ppDelta(-0.0016)).toBe("−0.160 pp");
    expect(criteo.prob(0.0528)).toBe("5.28%");
  });

  it("never emits a currency symbol", () => {
    const outputs = [criteo.pp(1), criteo.ppValue(1), criteo.ppDelta(1), criteo.prob(1)];
    for (const o of outputs) expect(o).not.toMatch(/[$€£]/);
  });

  it("exposes plain numeric conversion for chart axes", () => {
    expect(criteo.toPp(0.00674)).toBeCloseTo(0.674, 10);
    expect(hillstrom.toPercent(0.1819)).toBeCloseTo(18.19, 10);
  });
});

describe("generic formatters", () => {
  it("builds interval strings with one formatter", () => {
    expect(interval(0.674, 0.616, 0.715, (n) => n.toFixed(3))).toEqual({
      value: "0.674",
      range: "[0.616, 0.715]",
    });
  });

  it("formats p-values", () => {
    expect(pValue(0.00016)).toBe("p < 0.001");
    expect(pValue(0.8334, "Holm p")).toBe("Holm p = 0.833");
    expect(pValue(null)).toBe("n/a");
    expect(pValue(Number.NaN)).toBe("n/a");
  });

  it("formats counts, durations, memory and budgets", () => {
    expect(count(2795918)).toBe("2,795,918");
    expect(compact(2795918)).toBe("2.80M");
    expect(compact(12800)).toBe("12.8K");
    expect(compact(950)).toBe("950");
    expect(ms(0.33)).toBe("0.33 ms");
    expect(ms(20.43)).toBe("20.4 ms");
    expect(seconds(48.329)).toBe("48.3 s");
    expect(gigabytes(2.1496)).toBe("2.15 GB");
    expect(budget(0.2)).toBe("20%");
  });

  it("truncates hashes without altering short values", () => {
    const h = "1e5059cbf4ed40816157c9f6611dbf43348ffba7700885cb70b355fd65dcf980";
    expect(shortHash(h)).toBe("1e5059cb…dcf980");
    expect(shortHash("abc123")).toBe("abc123");
  });
});
