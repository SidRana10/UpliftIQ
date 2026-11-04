import { describe, expect, it } from "vitest";
import {
  criteoBudgetPoints,
  criteoInsight,
  criteoIpwCaveat,
  criteoKpis,
  criteoTreatAll,
  hillstromInsight,
  hillstromKpis,
  hillstromOutcomeRows,
  hillstromPolicyRows,
} from "@/features/overview/select";
import { criteoHeldout, hillstromDecision, overviewHillstrom } from "./fixtures";

describe("Hillstrom overview view-models", () => {
  it("builds four KPIs with the frozen decision first and Holm-supported lift", () => {
    const k = hillstromKpis(overviewHillstrom);
    expect(k).toHaveLength(4);
    expect(k[0]).toMatchObject({ label: "Frozen decision", value: "Men's E-Mail for all" });
    expect(k[1]).toMatchObject({ value: "18.2%", range: "[17.1%, 19.3%]", badge: "primary" });
    expect(k[2]).toMatchObject({ value: "+2.94 pp", badge: "supported" });
    expect(k[2].note).toContain("Holm p < 0.001");
    expect(k[3].value).toBe("12,800");
  });

  it("does not mark the lift as supported when Holm p is not below 0.05", () => {
    const weak = {
      ...overviewHillstrom,
      vs_strongest_alternative_baseline: { ...overviewHillstrom.vs_strongest_alternative_baseline, p_value_holm: 0.4 },
    };
    expect(hillstromKpis(weak)[2].badge).toBeUndefined();
  });

  it("orders policies with the frozen one first and uses only AIPW visit rows", () => {
    const rows = hillstromPolicyRows(hillstromDecision);
    expect(rows.map((r) => r.key)).toEqual(["mens_for_all", "womens_for_all", "random", "no_email"]);
    expect(rows[0]).toMatchObject({ emphasis: true, hollow: false, valueLabel: "18.2%" });
    expect(rows[1].hollow).toBe(true);
    expect(rows[0].value).toBeCloseTo(18.19, 6); // percent units for the axis
    expect(rows.every((r) => r.low <= r.value && r.value <= r.high)).toBe(true);
  });

  it("keeps primary and exploratory outcomes apart", () => {
    const rows = hillstromOutcomeRows(hillstromDecision);
    expect(rows.map((r) => [r.outcome, r.role])).toEqual([
      ["visit", "primary"],
      ["conversion", "exploratory"],
      ["spend", "exploratory"],
    ]);
    expect(rows[0].supported).toBe(true);
    expect(rows[1].supported).toBe(false);
    expect(rows[2].supported).toBe(false); // exploratory outcomes are never "supported"
    expect(rows[0].diff).toBe("+2.94 pp");
    expect(rows[1].frozen).toBe("0.91%");
    expect(rows[2].frozen).toBe("$1.16");
    expect(rows[2].diff).toBe("−$0.11");
  });

  it("writes an insight from the data, flagging exploratory outcomes as exploratory", () => {
    const text = hillstromInsight(overviewHillstrom, hillstromOutcomeRows(hillstromDecision));
    expect(text.headline).toBe("Send the Men's E-Mail for all.");
    expect(text.paragraphs[0]).toContain("12,800 held-out customers");
    expect(text.paragraphs[0]).toContain("18.2%");
    const last = text.paragraphs.at(-1)!;
    expect(last.startsWith("Conversion and Spend are exploratory")).toBe(true);
    expect(last).toContain("neither shows a clear advantage");
  });
});

describe("Criteo overview view-models", () => {
  it("builds KPIs in percentage points from the numeric held-out rows", () => {
    const k = criteoKpis(criteoHeldout);
    expect(k).toHaveLength(4);
    expect(k[0]).toMatchObject({ value: "0.674 pp", range: "[0.616, 0.715]", badge: "primary" });
    expect(k[1].value).toBe("0.525 pp");
    expect(k[2]).toMatchObject({ value: "0.744 pp", range: "[0.687, 0.793]" });
    expect(k[3].value).toBe("2.80M");
    for (const kpi of k) expect(`${kpi.value}${kpi.range ?? ""}${kpi.note}`).not.toMatch(/[$€£]/);
  });

  it("returns only the evaluated AIPW budgets, never interpolated or IPW/uplift>0 rows", () => {
    const pts = criteoBudgetPoints(criteoHeldout);
    expect(pts.map((p) => p.budget)).toEqual([10, 20, 30]);
    expect(pts.map((p) => p.primary)).toEqual([false, true, false]);
    expect(pts[1].model.valueLabel).toBe("0.674");
    expect(pts[1].random.valueLabel).toBe("0.149");
    expect(pts[1].model.value).toBeCloseTo(0.6741, 4); // pp for the axis
    expect(pts[1].vsRandomLabel).toBe("0.525 pp [0.480, 0.558]");
  });

  it("uses the AIPW treat-all reference, not IPW", () => {
    const t = criteoTreatAll(criteoHeldout)!;
    expect(t.valueLabel).toBe("0.744");
    expect(t.rangeLabel).toBe("[0.687, 0.793] pp");
  });

  it("drops a budget whose random reference is missing rather than inventing it", () => {
    const partial = {
      ...criteoHeldout,
      numeric: {
        ...criteoHeldout.numeric,
        references: criteoHeldout.numeric.references.filter((r) => r.policy !== "random@20%"),
      },
    };
    expect(criteoBudgetPoints(partial).map((p) => p.budget)).toEqual([10, 30]);
  });

  it("returns nothing (not fabricated values) when the primary budget row is absent", () => {
    const none = {
      ...criteoHeldout,
      numeric: { ...criteoHeldout.numeric, policies: criteoHeldout.numeric.policies.filter((p) => p.budget_requested !== 0.2) },
    };
    expect(criteoKpis(none)).toEqual([]);
    expect(criteoInsight(none)).toBeNull();
  });

  it("states benchmark-sample scope and keeps the IPW/AIPW discrepancy visible", () => {
    const insight = criteoInsight(criteoHeldout)!;
    expect(insight.headline).toBe("Contact the top 20% by predicted uplift.");
    expect(insight.paragraphs[0]).toContain("released benchmark");
    expect(insight.paragraphs.join(" ")).not.toMatch(/ROI|revenue|profit|\$/);
    expect(insight.paragraphs[1]).toContain("only for the 10%, 20% and 30% budgets");
    const ipw = criteoIpwCaveat(criteoHeldout);
    expect(ipw).toContain("1.020 pp vs 0.744 pp");
    expect(ipw).toContain("AIPW is the primary estimator");
  });
});
