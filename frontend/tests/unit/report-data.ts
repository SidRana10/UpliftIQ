/* eslint-disable @typescript-eslint/no-explicit-any -- test-only helpers over untyped report JSON */
/**
 * TEST-ONLY: builds API-shaped payloads straight from the committed report files in ../reports,
 * mirroring how src/upliftiq/backend/routes_reports.py assembles them. Unit tests run the real
 * view-models against these, so chart values are checked against the reports themselves.
 */
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import type {
  CriteoBenchmarkData,
  CriteoDiagnosticsData,
  CriteoHeldoutData,
  CriteoHeldoutExtras,
  CriteoIpwAipwData,
  HillstromBalanceRow,
  HillstromEffectsData,
  HillstromGroupRow,
  HillstromModelsData,
} from "@/lib/api/types";

const ROOT = join(__dirname, "..", "..", "..", "reports");
export const reportsAvailable = existsSync(join(ROOT, "criteo", "benchmark_full", "final_results.json"));

export function readJson<T = any>(rel: string): T {
  // Python writes literal NaN; the API serializes it as null
  const text = readFileSync(join(ROOT, rel), "utf8").replace(/\bNaN\b/g, "null");
  return JSON.parse(text) as T;
}

function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = "";
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') {
        cell += '"';
        i++;
      } else if (ch === '"') quoted = false;
      else cell += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ",") {
      row.push(cell);
      cell = "";
    } else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[i + 1] === "\n") i++;
      row.push(cell);
      cell = "";
      if (row.length > 1 || row[0] !== "") rows.push(row);
      row = [];
    } else cell += ch;
  }
  if (cell !== "" || row.length) {
    row.push(cell);
    rows.push(row);
  }
  return rows;
}

function typed(v: string): unknown {
  if (v === "") return null;
  if (v === "True") return true;
  if (v === "False") return false;
  const n = Number(v);
  return Number.isNaN(n) ? v : n;
}

export function readCsv<T>(rel: string): T[] {
  const [head, ...body] = parseCsv(readFileSync(join(ROOT, rel), "utf8"));
  return body.map((r) => Object.fromEntries(head.map((h, i) => [h, typed(r[i] ?? "")]))) as T[];
}

// ------------------------------------------------------------------ Hillstrom

export const hillstromEffects = (): HillstromEffectsData => ({
  full_sample: readCsv("hillstrom/tables/effects.csv"),
  heldout: readCsv("final_evaluation/observed_effects.csv"),
});
export const hillstromGroups = () => readCsv<HillstromGroupRow>("hillstrom/tables/group_summary.csv");
export const hillstromBalance = () => readCsv<HillstromBalanceRow>("hillstrom/tables/balance.csv");
export const hillstromModels = (): HillstromModelsData => ({
  candidates: readCsv("model_selection/selection_summary.csv"),
  selection: readJson("model_selection/frozen_config.json").selection,
});
export const hillstromChosen = (): string => readJson("model_selection/frozen_config.json").chosen_policy.name;

// ------------------------------------------------------------------ Criteo

const MODEL_FIELDS = ["family", "config_key", "n_fit", "fit_seconds", "predict_val_seconds", "latency", "rss_gb_before_fit", "rss_gb_peak_during_fit", "peak_rss_gb", "omp_num_threads", "hyperparameters", "training", "libraries", "code", "uplift_mean", "uplift_std"];

export function criteoBenchmark(): CriteoBenchmarkData {
  const full = readJson("criteo/benchmark_full/final_results.json");
  const scaling = readJson("criteo/benchmark_2m/scaling_results.json");
  return {
    split: "validation (500K subset); the model was selected on these numbers",
    n_train: full.n_fit,
    tables: Object.fromEntries(
      ["value", "references", "qini", "pairwise", "engineering"].map((k) => [k, full.tables[k]]),
    ) as CriteoBenchmarkData["tables"],
    selection: full.selection,
    change_from_2m: full.changes_vs_base,
    learning_curve: { levels: scaling.levels, changes: scaling.changes, engineering: scaling.engineering },
    numeric: {
      policies: Object.values<any>(full.policies)[0],
      qini: Object.values<any>(full.qini)[0],
      pairwise: full.pairwise,
      // the API passes through a fixed subset of each manifest (routes_reports.MODEL_FIELDS)
      models: Object.fromEntries(
        Object.entries<any>(full.manifests).map(([name, m]) => [name, Object.fromEntries(MODEL_FIELDS.filter((k) => k in m).map((k) => [k, m[k]]))]),
      ) as CriteoBenchmarkData["numeric"]["models"],
    },
  };
}

export function criteoHeldout(): CriteoHeldoutData & CriteoHeldoutExtras {
  const final = readJson("criteo/final/final_results.json");
  return {
    model: final.model,
    n: final.n,
    primary_metric: "AIPW policy value at 20% contact budget",
    assumption_check: final.assumption_check,
    evaluation_record: readJson("criteo/final/evaluation_record.json"),
    exploratory_conversion: { table: final.tables.conversion, note: "exploratory only; never used for any decision" },
    numeric: { policies: final.policies, references: final.references, qini: final.qini },
  };
}

export const criteoIpwAipw = (): CriteoIpwAipwData => readJson("criteo/benchmark/ipw_aipw_diagnostic.json");

export const criteoDiagnostics = (): CriteoDiagnosticsData => ({
  dataset: readJson("criteo/dataset_summary.json"),
  treatment: readJson("criteo/treatment_diagnostics.json"),
  split: readJson("criteo/split_summary.json"),
});
