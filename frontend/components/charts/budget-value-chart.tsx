"use client";

import {
  CartesianGrid,
  ErrorBar,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { COLORS } from "@/lib/colors";
import { TooltipCard } from "./chart-tooltip";
import { CHART, useReducedMotion } from "./theme";

/** An estimate in percentage points, with pre-formatted labels. */
export interface Stat {
  value: number;
  low: number;
  high: number;
  valueLabel: string;
  rangeLabel: string;
}

export interface BudgetPoint {
  /** contact budget in percent (10, 20, 30) */
  budget: number;
  model: Stat;
  random: Stat;
  /** model minus random at the same budget (paired), pre-formatted */
  vsRandomLabel: string;
  /** true for the pre-specified primary budget */
  primary?: boolean;
}

interface Pt {
  x: number;
  y: number;
  err: [number, number];
  stat: Stat;
  series: "model" | "random";
  budget: number;
  vsRandomLabel: string;
  primary?: boolean;
}

interface ShapeProps {
  cx?: number;
  cy?: number;
  payload?: Pt;
}

function Marker({ cx = 0, cy = 0, payload }: ShapeProps) {
  if (!payload) return null;
  const model = payload.series === "model";
  const color = model ? COLORS.teal : COLORS.ref;
  return (
    <g>
      <circle cx={cx} cy={cy} r={14} fill="transparent" />
      <circle
        cx={cx}
        cy={cy}
        r={model ? (payload.primary ? 6.5 : 5.5) : 4.5}
        fill={model ? color : "#fff"}
        stroke={color}
        strokeWidth={model ? 1.5 : 2}
      />
      <text
        x={cx + (model ? -11 : 11)}
        y={cy + 4}
        textAnchor={model ? "end" : "start"}
        fontSize={12}
        fontWeight={model ? 600 : 500}
        fill={model ? COLORS.text : COLORS.axisTitle}
      >
        {payload.stat.valueLabel}
      </text>
    </g>
  );
}

const OFFSET = 1.6; // visual offset of the two series around each budget tick (x is in % units)

/**
 * Held-out policy value at the evaluated contact budgets only. Markers + 95% whiskers, no line
 * joining them (values between budgets were never estimated). References: zero baseline, the
 * budget-matched random policy, and the unconstrained treat-all estimate with its interval.
 */
export function BudgetValueChart({
  points,
  treatAll,
  treatAllLabel,
  yTick,
  height = CHART.height.primary,
}: {
  points: BudgetPoint[];
  treatAll: Stat;
  treatAllLabel: string;
  yTick: (v: number) => string;
  height?: number;
}) {
  const animate = !useReducedMotion();
  const budgets = points.map((p) => p.budget);
  const xDomain: [number, number] = [Math.min(...budgets) - 6, Math.max(...budgets) + 6];
  const top = Math.max(treatAll.high, ...points.map((p) => p.model.high));
  const yMax = Math.ceil(top * 1.12 * 5) / 5; // round up to 0.2 pp

  const data = (series: "model" | "random"): Pt[] =>
    points.map((p) => {
      const stat = p[series];
      return {
        x: p.budget + (series === "model" ? -OFFSET : OFFSET),
        y: stat.value,
        err: [stat.value - stat.low, stat.high - stat.value],
        stat,
        series,
        budget: p.budget,
        vsRandomLabel: p.vsRandomLabel,
        primary: p.primary,
      };
    });

  return (
    <div
      role="img"
      aria-label={`Held-out policy value at ${budgets.map((b) => `${b}%`).join(", ")} contact budgets, with random and treat-all references`}
    >
      <ResponsiveContainer width="100%" height={height}>
        <ScatterChart margin={{ top: 14, right: 20, bottom: 4, left: 0 }}>
          <CartesianGrid vertical={false} stroke={COLORS.grid} />
          <XAxis
            type="number"
            dataKey="x"
            domain={xDomain}
            ticks={budgets}
            tickFormatter={(v: number) => `${v}%`}
            tick={CHART.axisTick}
            axisLine={CHART.axisLine}
            tickLine={false}
            tickMargin={6}
            height={44}
            label={{ value: "Contact budget (share of customers treated)", position: "insideBottom", offset: -2, fontSize: 12, fill: COLORS.axisTitle }}
          />
          <YAxis
            type="number"
            dataKey="y"
            domain={[0, yMax]}
            tickFormatter={yTick}
            tick={CHART.axisTick}
            axisLine={false}
            tickLine={false}
            width={44}
            label={{ value: "AIPW policy value (pp)", angle: -90, position: "insideLeft", offset: 8, fontSize: 12, fill: COLORS.axisTitle, style: { textAnchor: "middle" } }}
          />
          <ReferenceArea y1={treatAll.low} y2={treatAll.high} fill={COLORS.ref} fillOpacity={0.12} ifOverflow="visible" />
          <ReferenceLine y={treatAll.value} stroke={COLORS.ref} strokeDasharray="5 4" ifOverflow="visible" label={{ value: treatAllLabel, position: "insideTopLeft", fontSize: 11, fill: COLORS.axisTitle, offset: 6 }} />
          <ReferenceLine y={0} stroke={COLORS.zero} />
          <Tooltip
            cursor={false}
            isAnimationActive={false}
            content={({ active, payload }) => {
              const p = payload?.[0]?.payload as Pt | undefined;
              if (!active || !p) return null;
              const model = p.series === "model";
              return (
                <TooltipCard
                  title={model ? `S-learner, top ${p.budget}%` : `Random targeting, ${p.budget}%`}
                  color={model ? COLORS.teal : COLORS.ref}
                  rows={[
                    { label: "AIPW value", value: p.stat.valueLabel },
                    { label: "95% interval", value: p.stat.rangeLabel },
                    ...(model ? [{ label: "Minus random (paired)", value: p.vsRandomLabel }] : []),
                  ]}
                  footnote="Held-out estimate for the released Criteo sample."
                />
              );
            }}
          />
          <Scatter data={data("random")} shape={(p: ShapeProps) => <Marker {...p} />} isAnimationActive={animate} animationDuration={300}>
            <ErrorBar isAnimationActive={animate} dataKey="err" direction="y" width={9} stroke={COLORS.ref} strokeWidth={1.5} />
          </Scatter>
          <Scatter data={data("model")} shape={(p: ShapeProps) => <Marker {...p} />} isAnimationActive={animate} animationDuration={300}>
            <ErrorBar isAnimationActive={animate} dataKey="err" direction="y" width={9} stroke={COLORS.teal} strokeWidth={2} />
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
