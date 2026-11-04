import { describe, expect, it } from "vitest";
import {
  budgetRefs,
  budgetSeries,
  candidateRows,
  fmtDelta,
  fmtEst,
  heldoutQini,
  heldoutRows,
  hillstromDiffRows,
  hillstromSelectionSummary,
  latencyRows,
  leaderboard,
  modelDetail,
  modelSeries,
  pairReferences,
  pairwiseRows,
  resourceRows,
  scalingRows,
  selectionModel,
  sortLeaderboard,
  validationVsHeldout,
} from "@/features/uplift/select";
import * as R from "./report-data";

const bench = R.criteoBenchmark();
const held = R.criteoHeldout();

describe("estimate formatting", () => {
  it("fmtEst: pp with an interval; fmtDelta: signed, unit on the value only", () => {
    expect(fmtEst({ value: 0.00748, low: 0.00635, high: 0.0085 })).toMatchObject({ value: "0.748 pp", range: "[0.635, 0.850]" });
    expect(fmtDelta({ value: 0.00587, low: 0.00497, high: 0.00668 })).toMatchObject({
      value: "+0.587 pp",
      range: "[+0.497, +0.668]",
      text: "+0.587 [+0.497, +0.668] pp",
    });
    expect(fmtDelta({ value: -0.0008, low: -0.0015, high: -0.0001 }).value).toBe("−0.080 pp");
  });
});

describe.skipIf(!R.reportsAvailable)("Uplift Explorer, Criteo validation (against committed reports)", () => {
  it("learning curve: six models through 2M, three finalists with one more full-scale point", () => {
    const series = modelSeries(bench);
    expect(series.map((s) => s.model)).toEqual(["logistic", "s_learner", "t_learner", "dr_learner", "tarnet_cpu", "tarnet_mps"]);
    for (const s of series) {
      const sizes = s.points.map((p) => p.size);
      if (s.finalist) {
        expect(sizes).toEqual([100_000, 500_000, 2_000_000, bench.n_train]);
        expect(s.points.map((p) => p.fullScale)).toEqual([false, false, false, true]);
      } else {
        expect(sizes).toEqual([100_000, 500_000, 2_000_000]);
        expect(s.points.some((p) => p.fullScale)).toBe(false);
      }
    }
  });

  it("learning-curve points equal the stored scaling levels and full-scale policy rows", () => {
    const s = modelSeries(bench).find((x) => x.model === "s_learner")!;
    const lvl = bench.learning_curve.levels.find((l) => l.model === "s_learner" && l.size === 2_000_000)!;
    expect(s.points[2].est.v20).toEqual({ value: lvl["value@20%"][0], low: lvl["value@20%"][1], high: lvl["value@20%"][2] });
    const row = bench.numeric.policies.find((p) => p.model === "s_learner" && p.estimator === "AIPW" && p.budget_requested === 0.2)!;
    expect(s.points[3].est.v20).toEqual({ value: row.value, low: row.ci_low, high: row.ci_high });
    expect(s.points[3].trainSeconds).toBe(bench.numeric.models.s_learner.fit_seconds);
    expect(s.points[0].trainSeconds).toBeCloseTo(
      bench.learning_curve.engineering.find((e) => e.model === "s_learner" && e.size === 100_000)!.train_s,
      12,
    );
  });

  it("2M to full-scale changes line up with the stored paired change", () => {
    const rows = scalingRows(bench, "v20", ["s_learner", "t_learner"]);
    expect(rows.map((r) => r.model)).toEqual(["s_learner"]); // T-learner has no full-scale run
    expect(rows[0].steps.map((s) => s.step)).toEqual(["100K → 500K", "500K → 2M", "2M → full"]);
    const stored = bench.change_from_2m.find((c) => c.model === "s_learner" && c.metric === "value@20%")!;
    expect(rows[0].steps[2].est).toEqual({ value: stored.change[0], low: stored.change[1], high: stored.change[2] });
    expect(rows[0].statement).toMatch(/includes zero/);
  });

  it("leaderboard: three finalists with the selection flags from the record", () => {
    const rows = leaderboard(bench);
    expect(rows.map((r) => r.model).sort()).toEqual(["logistic", "s_learner", "tarnet_cpu"]);
    const s = rows.find((r) => r.model === "s_learner")!;
    expect(s).toMatchObject({ selected: true, bestByQini: true, bestByValue: false });
    expect(rows.find((r) => r.model === "tarnet_cpu")!.bestByValue).toBe(true);
    expect(fmtEst(s.v20).value).toBe("0.748 pp");
    expect(fmtDelta(s.vsRandom20).value).toBe("+0.587 pp");
    expect(fmtEst(s.qini).value).toBe("0.335 pp");
    expect(s.trainSeconds).toBe(bench.numeric.models.s_learner.fit_seconds);
    expect(s.rowsPerS).toBe(bench.numeric.models.s_learner.latency.find((l) => l.batch === 100000)!.rows_per_s);
  });

  it("leaderboard sorts by any column in both directions, missing values last", () => {
    const rows = leaderboard(bench);
    expect(sortLeaderboard(rows, "v20", "desc")[0].model).not.toBe("logistic");
    expect(sortLeaderboard(rows, "trainSeconds", "asc").map((r) => r.model)).toEqual(["logistic", "s_learner", "tarnet_cpu"]);
    expect(sortLeaderboard(rows, "trainSeconds", "desc")[0].model).toBe("tarnet_cpu");
    expect(sortLeaderboard(rows, "label", "asc").map((r) => r.label)).toEqual(["Logistic", "S-learner", "TARNet (CPU)"]);
    const withGap = [{ ...rows[0], p50Ms1: NaN }, rows[1], rows[2]];
    expect(sortLeaderboard(withGap, "p50Ms1", "asc").at(-1)!.model).toBe(rows[0].model);
    expect(sortLeaderboard(withGap, "p50Ms1", "desc").at(-1)!.model).toBe(rows[0].model);
  });

  it("budget series: evaluated budgets only, finalists only, no interpolated points", () => {
    const s = budgetSeries(bench, ["s_learner", "t_learner", "logistic"]);
    expect(s.map((x) => x.model)).toEqual(["s_learner", "logistic"]);
    for (const x of s) expect(x.points.map((p) => p.budget)).toEqual([10, 20, 30]);
    expect(s[0].points[1].label).toBe("0.748 pp");
    expect(budgetSeries(bench, ["t_learner"])).toEqual([]);
  });

  it("budget references: exact stored random values and the report's own treat-all text", () => {
    const r = budgetRefs(bench);
    expect(r.random.map((x) => x.budget)).toEqual([10, 20, 30]);
    const row = bench.numeric.policies.find((p) => p.estimator === "AIPW" && p.budget_requested === 0.2)!;
    expect(r.random[1].value).toBe(row.random_value);
    expect(r.random[1].reportText).toBe(bench.tables.references.find((x) => x.policy === "random@20%")!["AIPW value (pp)"]);
    expect(r.treatAll?.reportText).toBe("0.808 [0.677, 0.923]");
  });

  it("paired differences use the stored AIPW rows for the chosen reference and metric", () => {
    expect(pairReferences(bench).sort()).toEqual(["logistic", "tarnet_cpu"]);
    const rows = pairwiseRows(bench, "tarnet_cpu", "v20");
    expect(rows.map((r) => r.model).sort()).toEqual(["logistic", "s_learner"]);
    const logistic = rows.find((r) => r.model === "logistic")!;
    const stored = bench.numeric.pairwise.tarnet_cpu.find((r) => r.model === "logistic" && r.estimator === "AIPW" && r.metric === "value@20%")!;
    expect(logistic.est).toEqual({ value: stored.diff, low: stored.ci_low, high: stored.ci_high });
    expect(logistic.excludesZero).toBe(true);
    expect(rows.find((r) => r.model === "s_learner")!.excludesZero).toBe(false);
    // the metric selector switches the stored rows
    expect(pairwiseRows(bench, "logistic", "qini")[0].est.value).not.toBe(pairwiseRows(bench, "logistic", "v20")[0].est.value);
    expect(pairwiseRows(bench, "nope", "v20")).toEqual([]);
  });

  it("selection record: simplicity order, both steps, chosen model and the Qini cross-check", () => {
    const s = selectionModel(bench);
    expect(s.order.map((o) => o.model)).toEqual(["logistic", "s_learner", "tarnet_cpu"]);
    expect(s.steps).toHaveLength(2);
    expect(s.steps[0]).toMatchObject({ simpler: "logistic", best: "tarnet_cpu", withinOneSe: false });
    expect(s.steps[1]).toMatchObject({ simpler: "s_learner", best: "tarnet_cpu", withinOneSe: true });
    expect(s.steps[1].diff).toBe("0.000 pp [−0.044, +0.047]");
    expect(s.chosen).toBe("s_learner");
    expect(s.qiniSupports).toBe(true);
    expect(s.qini.map((q) => q.model)).toEqual(["logistic", "s_learner", "tarnet_cpu"]);
    expect(s.summary.join(" ")).toContain("simplest eligible model");
  });

  it("engineering: latency, resources and model card come from the stored manifests", () => {
    const lat = latencyRows(bench, ["s_learner", "t_learner", "tarnet_cpu"]);
    expect(lat.map((l) => l.model)).toEqual(["s_learner", "tarnet_cpu"]);
    expect(lat[0].batches.map((b) => b.batch)).toEqual([1, 1000, 100000]);
    const res = resourceRows(bench, ["s_learner"])[0];
    expect(res.trainSeconds).toBe("48.3 s");
    expect(res.peakFit).toBe("2.15 GB");
    const card = modelDetail("tarnet_cpu", bench.numeric.models.tarnet_cpu);
    expect(card.training.find((t) => t.label === "Epochs run")?.value).toBe("15");
    expect(card.dirty).toBe(true);
    expect(card.hyperparameters.some((h) => h.label === "arch.hidden")).toBe(true);
  });
});

describe.skipIf(!R.reportsAvailable)("Uplift Explorer, Criteo held-out (against committed reports)", () => {
  it("held-out rows: the three evaluated budgets with IPW as a muted cross-check", () => {
    const rows = heldoutRows(held);
    expect(rows.map((r) => r.budget)).toEqual(["10%", "20%", "30%"]);
    expect(rows[1]).toMatchObject({ value: "0.674 pp", range: "[0.616, 0.715]", ipw: "0.969 pp", primary: true });
    expect(rows.filter((r) => r.primary)).toHaveLength(1);
    expect(heldoutQini(held).aipw?.value).toBe("0.318 pp");
  });

  it("validation vs held-out for the selected model, at the evaluated budgets only", () => {
    const rows = validationVsHeldout(bench, held);
    expect(rows.map((r) => r.budget)).toEqual(["10%", "20%", "30%"]);
    expect(rows[1].validation).toBe("0.748 [0.635, 0.850] pp");
    expect(rows[1].heldout).toBe("0.674 [0.616, 0.715] pp");
  });
});

describe.skipIf(!R.reportsAvailable)("Uplift Explorer, Hillstrom (against committed reports)", () => {
  const models = R.hillstromModels();
  const chosen = R.hillstromChosen();

  it("eight candidates: four models and four fixed baselines, Men's for all selected", () => {
    const rows = candidateRows(models, chosen);
    expect(rows).toHaveLength(8);
    expect(rows.filter((r) => r.kind === "model")).toHaveLength(4);
    expect(rows.filter((r) => r.chosen).map((r) => r.policy)).toEqual(["mens_for_all"]);
    expect(rows.find((r) => r.policy === "t_learner")).toMatchObject({ eligible: true, bestByValue: true });
    expect(rows.find((r) => r.policy === "s_learner")!.eligible).toBe(false);
    expect(rows.find((r) => r.policy === "mens_for_all")!.valueLabel).toBe("18.26%");
    expect(rows.find((r) => r.policy === "random")!.vsMens).toBeNull();
  });

  it("paired differences vs Men's for all: four models, T-learner interval includes zero", () => {
    const d = hillstromDiffRows(models);
    expect(d.map((x) => x.key)).toEqual(["logistic", "s_learner", "t_learner", "dr_learner"]);
    const t = d.find((x) => x.key === "t_learner")!;
    expect(t.valueLabel).toBe("+0.12 pp");
    expect(t.rangeLabel).toBe("[−0.24 pp, +0.49 pp]");
    expect(d.every((x) => x.includesZero)).toBe(true);
  });

  it("selection summary reflects the stored selection and the zero-crossing interval", () => {
    const t = hillstromSelectionSummary(models, chosen).join(" ");
    expect(t).toContain("Selected policy: Men's e-mail for all");
    expect(t).toContain("do not show that personalization beats");
    expect(t).toContain("Eligible personalized models (not below the best fixed baseline): LightGBM T-learner");
  });
});
