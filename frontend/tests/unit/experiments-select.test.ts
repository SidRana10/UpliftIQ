import { describe, expect, it } from "vitest";
import {
  armBars,
  armSummary,
  balanceRows,
  checkRows,
  comparisonKey,
  criteoLabKpis,
  effectReadout,
  estimatorRows,
  forestRows,
  gapModel,
  heldoutRepeat,
  significance,
  smdRows,
  strataPoints,
} from "@/features/experiments/select";
import * as R from "./report-data";

const effects = R.hillstromEffects();
const heldout = (o: string, k: "mens" | "womens") =>
  effects.heldout.find((r) => r.outcome === o && comparisonKey(r) === k)!;

describe.skipIf(!R.reportsAvailable)("Hillstrom Experiment Lab view-models (against committed reports)", () => {
  it("derives significance from the Holm flag and the outcome role, never from the outcome alone", () => {
    expect(significance(heldout("visit", "mens"), "primary")).toBe("supported");
    // significant but exploratory stays labelled exploratory
    expect(significance(heldout("spend", "womens"), "exploratory")).toBe("exploratory-significant");
    // not Holm-significant on the held-out sample, even though its unadjusted CI excludes zero
    const mensSpend = heldout("spend", "mens");
    expect(mensSpend.abs_ci_low).toBeGreaterThan(0);
    expect(significance(mensSpend, "exploratory")).toBe("not-significant");
    expect(significance(heldout("conversion", "mens"), "exploratory")).toBe("not-significant");
  });

  it("full sample and held-out disagree on significance for the same comparison", () => {
    const full = effects.full_sample.find((r) => r.outcome === "spend" && comparisonKey(r) === "mens")!;
    expect(significance(full, "exploratory")).toBe("exploratory-significant");
    expect(significance(heldout("spend", "mens"), "exploratory")).toBe("not-significant");
  });

  it("forest rows: one per comparison, rate effects in pp, spend in dollars, CI carried through", () => {
    const visit = forestRows(effects.heldout, "visit", "absolute", "primary");
    expect(visit).toHaveLength(2);
    const m = heldout("visit", "mens");
    expect(visit[0].value).toBeCloseTo(m.abs_effect * 100, 10);
    expect(visit[0].low).toBeCloseTo(m.abs_ci_low * 100, 10);
    expect(visit[0].high).toBeCloseTo(m.abs_ci_high * 100, 10);
    expect(visit[0].valueLabel).toBe("+7.67 pp");
    expect(visit.every((r) => r.significance === "supported")).toBe(true);

    const spend = forestRows(effects.heldout, "spend", "absolute", "exploratory");
    expect(spend[0].value).toBeCloseTo(heldout("spend", "mens").abs_effect, 10);
    expect(spend[0].valueLabel).toBe("+$0.64");
    expect(spend.map((r) => r.significance)).toEqual(["not-significant", "exploratory-significant"]);
  });

  it("relative scale uses the stored relative effect and interval", () => {
    const rel = forestRows(effects.heldout, "visit", "relative", "primary")[0];
    const m = heldout("visit", "mens");
    expect(rel.value).toBeCloseTo(m.rel_effect * 100, 10);
    expect(rel.low).toBeCloseTo(m.rel_ci_low * 100, 10);
    expect(rel.valueLabel).toBe("+72%");
  });

  it("arm bars come from the selected sample and agree with the group summary on the full sample", () => {
    const groups = R.hillstromGroups();
    const bars = armBars(effects.full_sample, "visit");
    expect(bars.map((b) => b.key)).toEqual(["control", "mens", "womens"]);
    const ctrl = groups.find((g) => g.segment === "No E-Mail")!;
    expect(bars[0].value).toBeCloseTo(ctrl.visit_rate * 100, 8);
    expect(bars[0].n).toBe(ctrl.n);
    const heldBars = armBars(effects.heldout, "visit");
    expect(heldBars[0].n).toBe(heldout("visit", "mens").n_control);
    expect(heldBars[0].value).not.toBeCloseTo(bars[0].value, 3);
  });

  it("arm summary adds zero-spend share only for the full sample", () => {
    const groups = R.hillstromGroups();
    const full = armSummary(effects.full_sample, "full", groups);
    expect(full).toHaveLength(3);
    expect(full[0].zeroSpend).toBe("99.4%");
    expect(armSummary(effects.heldout, "heldout", groups)[0].zeroSpend).toBeUndefined();
  });

  it("readout is built from the stored numbers and keeps the exploratory caveat", () => {
    const r = effectReadout(heldout("visit", "mens"), "primary");
    expect(r.significance).toBe("supported");
    expect(r.headline).toBe("Men's e-mail: +7.67 pp");
    expect(r.paragraphs.join(" ")).toContain("primary outcome");
    const s = effectReadout(heldout("spend", "womens"), "exploratory");
    expect(s.paragraphs.join(" ")).toMatch(/exploratory outcome and was not used for the campaign decision/);
    const ns = effectReadout(heldout("spend", "mens"), "exploratory");
    expect(ns.paragraphs.join(" ")).toContain("Not significant after Holm correction");
    expect(ns.paragraphs.join(" ")).not.toMatch(/interval includes zero/);
  });

  it("balance rows carry both arms' max |SMD| from the report", () => {
    const rows = balanceRows(R.hillstromBalance());
    expect(rows).toHaveLength(8);
    const recency = rows.find((r) => r.key === "recency")!;
    expect(recency.values.mens).toBeCloseTo(0.0068315325, 8);
    expect(recency.labels.womens).toBe("0.005");
  });
});

describe.skipIf(!R.reportsAvailable)("Criteo Experiment Lab view-models (against committed reports)", () => {
  const diag = R.criteoDiagnostics();
  const ipw = R.criteoIpwAipw();

  it("KPIs: rows, treated share and the benchmark verdict", () => {
    const k = criteoLabKpis(diag);
    expect(k).toHaveLength(4);
    expect(k[0].value).toBe("13.98M");
    expect(k[1].value).toBe("85.0%");
    expect(k[2].value).toBe("8.95M · 2.24M · 2.80M");
    expect(k[3].value).toBe("Modeling allowed");
  });

  it("randomization checks keep their grades, including both warnings", () => {
    const rows = checkRows(diag.treatment.checks);
    expect(rows.map((r) => r.id)).toEqual(["A1a", "A1b", "A1c", "A1d", "A2"]);
    expect(Object.fromEntries(rows.map((r) => [r.id, r.grade]))).toEqual({
      A1a: "pass",
      A1b: "pass",
      A1c: "warn",
      A1d: "ordering_warning",
      A2: "pass",
    });
    expect(rows[4].value).toBe("0.846 / 0.872");
  });

  it("SMD rows: 12 features, signed values, KS in the tooltip rows", () => {
    const rows = smdRows(diag);
    expect(rows).toHaveLength(12);
    const f0 = rows[0];
    expect(f0.values.smd).toBeCloseTo(diag.treatment.smd[0].smd, 12);
    expect(f0.extra?.find((e) => e.label === "KS statistic")?.value).toBe("0.009");
  });

  it("estimator rows: IPW, AIPW, g-computation and post-stratified, converted to pp", () => {
    const rows = estimatorRows(ipw);
    expect(rows.map((r) => r.key)).toEqual(["ipw", "aipw", "gcomp", "ps"]);
    expect(rows[0].value).toBeCloseTo(ipw.estimates.ipw * 100, 8);
    expect(rows[0].valueLabel).toBe("1.100 pp");
    expect(rows[1].valueLabel).toBe("0.808 pp");
    expect(rows[0].value).toBeGreaterThan(rows[1].value);
    // the post-stratified interval is explicitly a normal approximation
    expect(rows[3].tooltipNote).toMatch(/1\.96 SE/);
    const ps = ipw.poststratified_on_predicted_baseline;
    expect(rows[3].low).toBeCloseTo((ps.estimate - 1.96 * ps.se) * 100, 8);
  });

  it("gap model reports the baseline imbalance and permutation test as stored", () => {
    const g = gapModel(ipw)!;
    expect(g.baselineTreated).toBe("4.120%");
    expect(g.baselineControl).toBe("3.848%");
    expect(g.gap).toBe("+0.272 pp");
    expect(g.z).toBe("6.3");
    expect(g.p).toBe("p < 0.001");
    expect(g.difference.value).toBe("+0.292 pp");
  });

  it("deciles: ten strata, treated share rises with the predicted baseline", () => {
    const s = strataPoints(ipw);
    expect(s).toHaveLength(10);
    expect(s[9].treatedShare).toBeGreaterThan(s[0].treatedShare);
    expect(s[9].effect).toBeCloseTo(ipw.strata[9].effect * 100, 10);
  });

  it("held-out IPW vs AIPW repeat matches the held-out record", () => {
    const h = heldoutRepeat(R.criteoHeldout().assumption_check);
    expect(h).toEqual({ ipw: "1.020 pp", aipw: "0.744 pp", diff: "0.276 pp", se: "0.019 pp" });
  });
});
