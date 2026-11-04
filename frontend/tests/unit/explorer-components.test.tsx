import { act, cleanup, fireEvent, render as rtlRender, screen, within } from "@testing-library/react";
import type { ReactElement } from "react";
import { TooltipProvider } from "@/components/ui/tooltip";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ChartFrame } from "@/components/charts/chart-frame";
import { HillstromLab } from "@/features/experiments/hillstrom-lab";
import { CriteoExplorer } from "@/features/uplift/criteo-explorer";
import { Leaderboard } from "@/features/uplift/leaderboard";
import { leaderboard } from "@/features/uplift/select";
import { resetUrl } from "./nav-mock";
import * as R from "./report-data";

vi.mock("next/navigation", async () => {
  const { useSearchParamsMock } = await import("./nav-mock");
  return { useSearchParams: useSearchParamsMock, useRouter: () => ({ refresh: vi.fn(), replace: vi.fn() }), usePathname: () => "/" };
});

// the app shell provides the tooltip context
const render = (ui: ReactElement) => rtlRender(<TooltipProvider>{ui}</TooltipProvider>);

beforeEach(resetUrl);
afterEach(cleanup);

const search = () => new URLSearchParams(window.location.search);
const click = (el: HTMLElement) => act(() => void fireEvent.click(el));

describe("ChartFrame", () => {
  it("'View as table' swaps the chart for the table of the plotted values", () => {
    render(
      <ChartFrame
        title="Held-out value"
        subtitle="pp"
        chart={<div>the chart</div>}
        table={<table><tbody><tr><td>0.674</td></tr></tbody></table>}
        sources={["reports/x.json"]}
      />,
    );
    expect(screen.getByText("the chart")).toBeInTheDocument();
    click(screen.getByRole("button", { name: /view as table/i }));
    expect(screen.getByRole("cell", { name: "0.674" })).toBeInTheDocument();
    expect(screen.queryByText("the chart")).not.toBeInTheDocument();
    expect(screen.getByText(/reports\/x\.json/)).toBeInTheDocument();
  });
});

describe.skipIf(!R.reportsAvailable)("HillstromLab selectors", () => {
  const props = {
    effects: R.hillstromEffects(),
    groups: R.hillstromGroups(),
    balance: R.hillstromBalance(),
    roles: { visit: "primary", conversion: "exploratory", spend: "exploratory" } as const,
    sources: { effects: ["e.csv"], groups: ["g.csv"], balance: ["b.csv"] },
    caveats: [],
  };

  it("outcome and sample selectors drive every dependent panel and the URL", () => {
    render(<HillstromLab {...props} />);
    expect(screen.getByRole("heading", { name: "Visit by arm" })).toBeInTheDocument();
    expect(screen.getByText("Men's e-mail: +7.66 pp")).toBeInTheDocument(); // full sample
    expect(screen.getByText("64,000 customers")).toBeInTheDocument(); // all three arms

    click(screen.getByRole("radio", { name: "Spend" }));
    expect(search().get("outcome")).toBe("spend");
    expect(screen.getByRole("heading", { name: "Spend by arm" })).toBeInTheDocument();
    expect(screen.getByText("Spend is an exploratory outcome")).toBeInTheDocument();
    expect(screen.getByText("Men's e-mail: +$0.77 per customer")).toBeInTheDocument();

    click(screen.getByRole("radio", { name: "Held-out" }));
    expect(search().get("sample")).toBe("heldout");
    expect(screen.getByText("12,800 customers")).toBeInTheDocument();
    expect(screen.getByText("Men's e-mail: +$0.64 per customer")).toBeInTheDocument();
    // held-out Men's spend is not Holm-significant; the badge comes from the data
    expect(screen.getAllByText("Not significant").length).toBeGreaterThan(0);
    expect(screen.getByText(/Not significant after Holm correction/)).toBeInTheDocument();

    click(screen.getByRole("radio", { name: "Women's" }));
    expect(search().get("compare")).toBe("womens");
    expect(screen.getByText(/Significant \(exploratory\)/)).toBeInTheDocument();
  });

  it("restores the view from the URL on load", () => {
    window.history.replaceState(null, "", "/?outcome=conversion&sample=heldout&scale=relative");
    render(<HillstromLab {...props} />);
    expect(screen.getByRole("heading", { name: "Conversion by arm" })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "Held-out" })).toHaveAttribute("data-state", "on");
    expect(screen.getByRole("radio", { name: "Relative" })).toHaveAttribute("data-state", "on");
  });

  it("falls back safely for an invalid URL value", () => {
    window.history.replaceState(null, "", "/?outcome=revenue&sample=everything");
    render(<HillstromLab {...props} />);
    expect(screen.getByRole("heading", { name: "Visit by arm" })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "Full sample" })).toHaveAttribute("data-state", "on");
  });

  it("the effects table view lists every plotted comparison with Holm p-values", () => {
    render(<HillstromLab {...props} />);
    click(screen.getAllByRole("button", { name: /view as table/i })[1]);
    const table = screen.getByRole("table", { name: "Treatment effects" });
    expect(within(table).getAllByRole("row")).toHaveLength(3);
    expect(within(table).getAllByText(/Holm p < 0.001/).length).toBe(2);
    expect(within(table).getAllByText("Supported").length).toBe(2);
  });
});

describe.skipIf(!R.reportsAvailable)("Criteo Uplift Explorer", () => {
  const bench = R.criteoBenchmark();
  const node = <div>held-out content</div>;

  it("defaults to the three finalists; at most three models; the URL tracks the selection", () => {
    render(<CriteoExplorer bench={bench} benchSources={["b.json"]} heldout={node} />);
    const group = screen.getByRole("group", { name: "Compared models" });
    const pressed = () => within(group).getAllByRole("button", { pressed: true }).map((b) => b.textContent?.replace(/≤2M/, "").trim());
    expect(pressed().sort()).toEqual(["Logistic", "S-learner", "TARNet (CPU)"]);
    // a fourth model cannot be added while three are selected
    expect(within(group).getByRole("button", { name: /T-learner/ })).toBeDisabled();

    click(within(group).getByRole("button", { name: /Logistic/ }));
    expect(search().get("models")).toBe("s_learner,tarnet_cpu");
    click(within(group).getByRole("button", { name: /T-learner/ }));
    expect(search().get("models")).toBe("s_learner,tarnet_cpu,t_learner");
    // the last remaining model cannot be removed
    click(within(group).getByRole("button", { name: /T-learner/ }));
    click(within(group).getByRole("button", { name: /TARNet \(CPU\)/ }));
    click(within(group).getByRole("button", { name: /S-learner/ }));
    expect(search().get("models")).toBe("s_learner");
  });

  it("metric and scope selectors write the URL; held-out scope shows only the frozen S-learner", () => {
    render(<CriteoExplorer bench={bench} benchSources={["b.json"]} heldout={node} />);
    click(screen.getByRole("radio", { name: "Qini" }));
    expect(search().get("metric")).toBe("qini");
    expect(screen.getAllByText(/Qini/).length).toBeGreaterThan(0);

    expect(screen.queryByText("held-out content")).not.toBeInTheDocument();
    click(screen.getByRole("radio", { name: "Held-out" }));
    expect(search().get("scope")).toBe("heldout");
    expect(screen.getByText("held-out content")).toBeInTheDocument();
    expect(screen.getByText("Frozen S-learner only")).toBeInTheDocument();
    expect(screen.queryByRole("group", { name: "Compared models" })).not.toBeInTheDocument();
    expect(screen.getByText(/only the frozen S-learner was scored on the untouched test set/i)).toBeInTheDocument();
    // validation content is not mixed into the held-out scope
    expect(screen.queryByRole("heading", { name: "Model leaderboard" })).not.toBeInTheDocument();
  });

  it("scope, metric and models are restored from the URL", () => {
    window.history.replaceState(null, "", "/?scope=validation&metric=v10&models=logistic");
    render(<CriteoExplorer bench={bench} benchSources={["b.json"]} heldout={node} />);
    expect(screen.getByRole("radio", { name: "AIPW @10%" })).toHaveAttribute("data-state", "on");
    const group = screen.getByRole("group", { name: "Compared models" });
    expect(within(group).getAllByRole("button", { pressed: true })).toHaveLength(1);
  });

  it("the selection stepper renders the stored steps", () => {
    render(<CriteoExplorer bench={bench} benchSources={["b.json"]} heldout={node} />);
    expect(screen.getByRole("heading", { name: "Why the S-learner was selected" })).toBeInTheDocument();
    expect(screen.getByText("More than one SE behind")).toBeInTheDocument();
    expect(screen.getByText("Within one SE")).toBeInTheDocument();
  });
});

describe.skipIf(!R.reportsAvailable)("Leaderboard", () => {
  const rows = leaderboard(R.criteoBenchmark());
  const names = () =>
    screen
      .getAllByRole("rowheader")
      .map((h) => h.textContent ?? "")
      .map((t) => (t.startsWith("TARNet") ? "TARNet" : t.startsWith("S-learner") ? "S-learner" : "Logistic"));

  it("sorts by any column and toggles direction; header exposes aria-sort", () => {
    const onOpen = vi.fn();
    render(<Leaderboard rows={rows} compared={["s_learner"]} metric="v20" onOpen={onOpen} sources={["b.json"]} trainRows={8946939} />);
    const th = screen.getByRole("columnheader", { name: /AIPW @20%/ });
    expect(th).toHaveAttribute("aria-sort", "descending");

    click(screen.getByRole("button", { name: /^Train/ }));
    expect(names()).toEqual(["Logistic", "S-learner", "TARNet"]); // fastest first
    click(screen.getByRole("button", { name: /^Train/ }));
    expect(names()).toEqual(["TARNet", "S-learner", "Logistic"]);
    expect(screen.getByRole("columnheader", { name: /Train/ })).toHaveAttribute("aria-sort", "descending");
  });

  it("opens the model card for a row", () => {
    const onOpen = vi.fn();
    render(<Leaderboard rows={rows} compared={[]} metric="v20" onOpen={onOpen} sources={["b.json"]} trainRows={1} />);
    click(screen.getAllByRole("button", { name: "Open model card for S-learner" })[0]);
    expect(onOpen).toHaveBeenCalledWith("s_learner");
  });

  it("marks the selected model and the best-by badges", () => {
    render(<Leaderboard rows={rows} compared={[]} metric="v20" onOpen={vi.fn()} sources={["b.json"]} trainRows={1} />);
    expect(screen.getAllByText("Selected").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Best by value").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Best by Qini").length).toBeGreaterThan(0);
  });
});
