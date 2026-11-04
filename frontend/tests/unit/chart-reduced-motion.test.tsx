import { cleanup, render } from "@testing-library/react";
import { cloneElement, type ReactElement } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { BudgetValueChart } from "@/components/charts/budget-value-chart";
import { DotWhiskerChart } from "@/components/charts/dot-whisker-chart";

// jsdom has no layout: give ResponsiveContainer a fixed size so the chart actually renders
vi.mock("recharts", async (orig) => {
  const actual = await orig<typeof import("recharts")>();
  return {
    ...actual,
    ResponsiveContainer: ({ children, height }: { children: ReactElement<{ width?: number; height?: number }>; height: number }) =>
      cloneElement(children, { width: 640, height }),
  };
});

function setReducedMotion(reduce: boolean) {
  window.matchMedia = ((query: string) => ({
    matches: reduce && query.includes("prefers-reduced-motion"),
    media: query,
    addEventListener: () => {},
    removeEventListener: () => {},
  })) as unknown as typeof window.matchMedia;
}

afterEach(cleanup);
beforeEach(() => setReducedMotion(true));

const stat = (v: number) => ({ value: v, low: v - 0.1, high: v + 0.1, valueLabel: v.toFixed(2), rangeLabel: "[a, b]" });

/** Whisker segments must never be left at a zero scale (invisible) when animation is disabled. */
function hiddenWhiskers(container: HTMLElement) {
  return [...container.querySelectorAll<SVGElement>(".recharts-errorBar line")].filter((l) =>
    /scale[XY]?\(0\)/.test(l.style.transform),
  );
}

describe("95% whiskers with prefers-reduced-motion", () => {
  it("DotWhiskerChart draws every interval", () => {
    const { container } = render(
      <DotWhiskerChart
        rows={[
          { key: "a", label: "A", value: 1, low: 0.5, high: 1.5, color: "#117E74", valueLabel: "1.0", rangeLabel: "[0.5, 1.5]" },
          { key: "b", label: "B", value: 2, low: 1.5, high: 2.5, color: "#7467C9", valueLabel: "2.0", rangeLabel: "[1.5, 2.5]" },
        ]}
        tickFormat={(v) => String(v)}
        axisTitle="x"
      />,
    );
    expect(container.querySelectorAll(".recharts-errorBar line").length).toBeGreaterThan(0);
    expect(hiddenWhiskers(container)).toHaveLength(0);
  });

  it("BudgetValueChart draws every interval", () => {
    const { container } = render(
      <BudgetValueChart
        points={[10, 20, 30].map((b) => ({ budget: b, model: stat(b / 30), random: stat(b / 100), vsRandomLabel: "+0.1", primary: b === 20 }))}
        treatAll={stat(0.8)}
        treatAllLabel="Treat everyone"
        yTick={(v) => String(v)}
      />,
    );
    expect(container.querySelectorAll(".recharts-errorBar line").length).toBeGreaterThan(0);
    expect(hiddenWhiskers(container)).toHaveLength(0);
  });
});
