import { describe, expect, it } from "vitest";
import { WORKSPACES, hrefWithDataset, parseDataset, workspaceFor } from "@/lib/datasets";

const ws = (id: string) => WORKSPACES.find((w) => w.id === id)!;

describe("dataset-aware URL state", () => {
  it("lists the five workspaces in plan order", () => {
    expect(WORKSPACES.map((w) => w.number)).toEqual(["01", "02", "03", "04", "05"]);
    expect(WORKSPACES.map((w) => w.href)).toEqual(["/overview", "/experiments", "/uplift", "/decisions", "/operations"]);
  });

  it("falls back to each workspace's default dataset", () => {
    expect(parseDataset(undefined, ws("overview"))).toBe("hillstrom");
    expect(parseDataset(undefined, ws("uplift"))).toBe("criteo");
    expect(parseDataset(undefined, ws("decisions"))).toBe("criteo");
  });

  it("accepts valid values and rejects anything else", () => {
    expect(parseDataset("criteo", ws("overview"))).toBe("criteo");
    expect(parseDataset(["hillstrom", "criteo"], ws("uplift"))).toBe("hillstrom");
    expect(parseDataset("nope", ws("overview"))).toBe("hillstrom");
    expect(parseDataset("", ws("uplift"))).toBe("criteo");
  });

  it("carries the dataset in links only where the destination supports it", () => {
    expect(hrefWithDataset(ws("experiments"), "criteo")).toBe("/experiments?dataset=criteo");
    expect(hrefWithDataset(ws("operations"), "criteo")).toBe("/operations");
    expect(hrefWithDataset(ws("overview"), null)).toBe("/overview");
  });

  it("resolves a workspace from a pathname", () => {
    expect(workspaceFor("/uplift")?.id).toBe("uplift");
    expect(workspaceFor("/uplift/anything")?.id).toBe("uplift");
    expect(workspaceFor("/unknown")).toBeUndefined();
  });
});
