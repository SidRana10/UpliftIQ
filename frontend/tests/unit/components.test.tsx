import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { CaveatCallout } from "@/components/data/caveat-callout";
import { DataTable } from "@/components/data/data-table";
import { EvidenceBadge } from "@/components/data/evidence-badge";
import { KpiCard } from "@/components/data/kpi-card";
import { SectionError } from "@/components/states/section-error";

vi.mock("next/navigation", () => ({ useRouter: () => ({ refresh: vi.fn() }) }));

afterEach(cleanup);

describe("KpiCard", () => {
  it("shows the pre-formatted value, interval and note without altering them", () => {
    render(<KpiCard label="AIPW value @ 20% budget" value="0.674 pp" range="[0.616, 0.715]" note="Pre-specified primary metric." />);
    expect(screen.getByText("0.674 pp")).toBeInTheDocument();
    expect(screen.getByText(/95% CI \[0\.616, 0\.715\]/)).toBeInTheDocument();
    expect(screen.getByText("Pre-specified primary metric.")).toBeInTheDocument();
  });

  it("omits the interval line when there is none", () => {
    render(<KpiCard label="Held-out customers" value="12,800" />);
    expect(screen.queryByText(/95% CI/)).not.toBeInTheDocument();
  });
});

describe("SectionError", () => {
  it("explains a network failure and offers a retry", () => {
    render(<SectionError what="Headline metrics" error={{ kind: "network", status: 0, message: "Could not reach the API (fetch failed)." }} />);
    expect(screen.getByRole("alert")).toBeInTheDocument();
    expect(screen.getByText("The API is not reachable")).toBeInTheDocument();
    expect(screen.getByText(/Headline metrics\./)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /retry/i })).toBeInTheDocument();
  });

  it("explains a missing report with its path and no pointless retry", () => {
    render(
      <SectionError
        what="Held-out policy values"
        error={{ kind: "report_unavailable", status: 404, message: "report not available: reports/criteo/final/final_results.json" }}
      />,
    );
    expect(screen.getByText("This report isn't available in this deployment")).toBeInTheDocument();
    expect(screen.getByText(/reports\/criteo\/final\/final_results\.json/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /retry/i })).not.toBeInTheDocument();
  });

  it("explains an unavailable model", () => {
    render(<SectionError what="Scoring" error={{ kind: "model_unavailable", status: 503, message: "not loaded" }} />);
    expect(screen.getByText("The scoring model isn't loaded")).toBeInTheDocument();
  });
});

describe("EvidenceBadge, CaveatCallout, DataTable", () => {
  it("labels evidence kinds distinctly", () => {
    render(
      <>
        <EvidenceBadge kind="heldout" />
        <EvidenceBadge kind="validation" />
        <EvidenceBadge kind="exploratory" />
        <EvidenceBadge kind="synthetic" />
      </>,
    );
    for (const t of ["Held-out", "Validation", "Exploratory", "Synthetic"]) expect(screen.getByText(t)).toBeInTheDocument();
  });

  it("renders caveats as a labelled note", () => {
    render(<CaveatCallout title="Benchmark, not business ROI" items={["Conditional estimates for the released sample."]} />);
    expect(screen.getByRole("note")).toHaveTextContent("Benchmark, not business ROI");
    expect(screen.getByText("Conditional estimates for the released sample.")).toBeInTheDocument();
  });

  it("renders a table with headers and cells", () => {
    render(
      <DataTable
        caption="Budgets"
        rows={[{ b: "10%", v: "0.582" }, { b: "20%", v: "0.674" }]}
        rowKey={(r) => r.b}
        columns={[
          { key: "b", header: "Budget", cell: (r) => r.b },
          { key: "v", header: "Value", align: "right", cell: (r) => r.v },
        ]}
      />,
    );
    expect(screen.getByRole("columnheader", { name: "Budget" })).toBeInTheDocument();
    expect(screen.getAllByRole("row")).toHaveLength(3);
    expect(screen.getByRole("cell", { name: "0.674" })).toBeInTheDocument();
  });
});

describe("splitLabel (narrow chart axis labels)", () => {
  it("wraps long labels at the space nearest the middle and leaves short ones alone", async () => {
    const { splitLabel } = await import("@/components/charts/dot-whisker-chart");
    expect(splitLabel("Women's E-Mail for all")).toEqual(["Women's E-Mail", "for all"]);
    expect(splitLabel("Men's E-Mail for all")).toEqual(["Men's E-Mail", "for all"]);
    expect(splitLabel("No e-mail")).toEqual(["No e-mail"]);
    expect(splitLabel("Uniform random")).toEqual(["Uniform", "random"]);
  });
});
