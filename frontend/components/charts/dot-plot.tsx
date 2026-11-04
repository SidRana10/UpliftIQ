"use client";

import {
  CartesianGrid,
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

export interface DotSeries {
  key: string;
  label: string;
  color: string;
  hollow?: boolean;
}

export interface DotPlotRow {
  key: string;
  label: string;
  /** series key -> value (plot units) */
  values: Record<string, number>;
  /** series key -> pre-formatted value */
  labels: Record<string, string>;
  /** extra tooltip rows, pre-formatted */
  extra?: { label: string; value: string }[];
}

interface Pt {
  x: number;
  y: number;
  row: DotPlotRow;
  series: DotSeries;
}

interface ShapeProps {
  cx?: number;
  cy?: number;
  payload?: Pt;
}

function Dot({ cx = 0, cy = 0, payload }: ShapeProps) {
  if (!payload) return null;
  const { series } = payload;
  return (
    <g>
      <circle cx={cx} cy={cy} r={12} fill="transparent" />
      <circle
        cx={cx}
        cy={cy}
        r={4.5}
        fill={series.hollow ? "#fff" : series.color}
        stroke={series.color}
        strokeWidth={series.hollow ? 2 : 1.5}
      />
    </g>
  );
}

/**
 * Horizontal dot plot: one row per covariate, one dot per series, with threshold lines. Used for
 * covariate balance (standardized mean differences); no interval exists for these statistics.
 */
export function DotPlot({
  rows,
  series,
  domain,
  thresholds,
  band,
  tickFormat,
  axisTitle,
  describe,
  rowHeight = 30,
}: {
  rows: DotPlotRow[];
  series: DotSeries[];
  domain: [number, number];
  thresholds: { x: number; label?: string; strong?: boolean }[];
  /** shaded "acceptable" region */
  band?: [number, number];
  tickFormat: (v: number) => string;
  axisTitle: string;
  describe: string;
  rowHeight?: number;
}) {
  const animate = !useReducedMotion();
  const offsets = series.length > 1 ? series.map((_, i) => (i - (series.length - 1) / 2) * 0.28) : [0];
  const height = rows.length * rowHeight + 70;

  return (
    <div role="img" aria-label={describe}>
      <ResponsiveContainer width="100%" height={height}>
        <ScatterChart margin={{ top: 8, right: 20, bottom: 4, left: 0 }}>
          <CartesianGrid horizontal={false} stroke={COLORS.grid} />
          <XAxis
            type="number"
            dataKey="x"
            domain={domain}
            tickFormatter={tickFormat}
            tick={CHART.axisTick}
            axisLine={CHART.axisLine}
            tickLine={false}
            tickMargin={6}
            height={44}
            label={{ value: axisTitle, position: "insideBottom", offset: -4, fontSize: 12, fill: COLORS.axisTitle }}
          />
          <YAxis
            type="number"
            dataKey="y"
            domain={[-0.6, rows.length - 0.4]}
            ticks={rows.map((_, i) => i)}
            tickFormatter={(i: number) => rows[i]?.label ?? ""}
            reversed
            axisLine={false}
            tickLine={false}
            width={92}
            tick={{ fontSize: 12, fill: COLORS.text }}
            interval={0}
          />
          {band ? <ReferenceArea x1={band[0]} x2={band[1]} fill={COLORS.teal} fillOpacity={0.05} ifOverflow="hidden" /> : null}
          {thresholds.map((t) => (
            <ReferenceLine
              key={t.x}
              x={t.x}
              stroke={t.strong ? COLORS.amber : COLORS.refSoft}
              strokeDasharray="4 3"
              label={t.label ? { value: t.label, position: "insideTopRight", fontSize: 11, fill: COLORS.axis } : undefined}
            />
          ))}
          <ReferenceLine x={0} stroke={COLORS.zero} />
          <Tooltip
            cursor={false}
            isAnimationActive={false}
            content={({ active, payload }) => {
              const p = payload?.[0]?.payload as Pt | undefined;
              if (!active || !p) return null;
              return (
                <TooltipCard
                  title={p.row.label}
                  color={p.series.color}
                  rows={[
                    ...series.map((s) => ({ label: s.label, value: p.row.labels[s.key] ?? "n/a" })),
                    ...(p.row.extra ?? []),
                  ]}
                />
              );
            }}
          />
          {series.map((s, si) => (
            <Scatter
              key={s.key}
              data={rows.map((row, i) => ({ x: row.values[s.key], y: i + offsets[si], row, series: s }) satisfies Pt)}
              shape={(p: ShapeProps) => <Dot {...p} />}
              isAnimationActive={animate}
              animationDuration={300}
            />
          ))}
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
