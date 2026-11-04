"use client";

import {
  CartesianGrid,
  ErrorBar,
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
import { CHART, useIsNarrow, useReducedMotion } from "./theme";

/** One estimate with its 95% interval. Numbers are in plotting units; labels are pre-formatted. */
export interface DotRow {
  key: string;
  label: string;
  sublabel?: string;
  value: number;
  low: number;
  high: number;
  color: string;
  emphasis?: boolean;
  hollow?: boolean;
  /** pre-formatted, e.g. "18.2%" and "[17.1%, 19.3%]" */
  valueLabel: string;
  rangeLabel: string;
  tooltipNote?: string;
}

interface Pt {
  x: number;
  y: number;
  err: [number, number];
  row: DotRow;
}

interface ShapeProps {
  cx?: number;
  cy?: number;
  payload?: Pt;
}

function Marker({ cx = 0, cy = 0, payload }: ShapeProps) {
  if (!payload) return null;
  const { row } = payload;
  const r = row.emphasis ? 5.5 : 4.5;
  return (
    <g>
      <circle cx={cx} cy={cy} r={14} fill="transparent" />
      <circle
        cx={cx}
        cy={cy}
        r={r}
        fill={row.hollow ? "#fff" : row.color}
        stroke={row.color}
        strokeWidth={row.hollow ? 2 : 1.5}
      />
      <text
        x={cx}
        y={cy - 13}
        textAnchor="middle"
        fontSize={12}
        fontWeight={row.emphasis ? 600 : 500}
        fill={row.emphasis ? COLORS.text : COLORS.axisTitle}
      >
        {row.valueLabel}
      </text>
    </g>
  );
}

/** Split a long label at the space nearest its middle (narrow screens wrap instead of clipping). */
export function splitLabel(label: string): string[] {
  if (label.length <= 12 || !label.includes(" ")) return [label];
  const mid = label.length / 2;
  let best = -1;
  for (let i = 0; i < label.length; i++) {
    if (label[i] === " " && (best < 0 || Math.abs(i - mid) < Math.abs(best - mid))) best = i;
  }
  return [label.slice(0, best), label.slice(best + 1)];
}

function RowTick({
  x = 0,
  y = 0,
  payload,
  rows,
  narrow,
}: {
  x?: number | string;
  y?: number | string;
  payload?: { value: number };
  rows: DotRow[];
  narrow: boolean;
}) {
  const row = payload ? rows[payload.value] : undefined;
  if (!row) return null;
  const lines = narrow ? splitLabel(row.label) : [row.label];
  const showSub = Boolean(row.sublabel) && !narrow;
  const lineHeight = 14;
  // vertically centre the label block on the row (leave room for the sublabel when shown)
  const first = (showSub ? -3 : 4) - ((lines.length - 1) * lineHeight) / 2;
  return (
    <g transform={`translate(${Number(x) - 8},${Number(y)})`}>
      {lines.map((line, i) => (
        <text
          key={i}
          textAnchor="end"
          fontSize={12}
          fontWeight={row.emphasis ? 600 : 500}
          fill={COLORS.text}
          y={first + i * lineHeight}
        >
          {line}
        </text>
      ))}
      {showSub ? (
        <text textAnchor="end" fontSize={11} fill={COLORS.axis} y={11}>
          {row.sublabel}
        </text>
      ) : null}
    </g>
  );
}

/** Horizontal dot-and-whisker (forest-style) chart: one row per estimate with its interval. */
export function DotWhiskerChart({
  rows,
  domain,
  tickFormat,
  axisTitle,
  refLines = [],
  rowHeight = 62,
}: {
  rows: DotRow[];
  domain?: [number, number];
  tickFormat: (v: number) => string;
  axisTitle: string;
  refLines?: { x: number; label: string; color?: string }[];
  rowHeight?: number;
}) {
  const animate = !useReducedMotion();
  const narrow = useIsNarrow();
  const lo = Math.min(...rows.map((r) => r.low));
  const hi = Math.max(...rows.map((r) => r.high));
  const pad = (hi - lo) * 0.14 || 1;
  const dom: [number, number] = domain ?? [Math.max(0, lo - pad), hi + pad];
  const height = rows.length * rowHeight + 56;

  return (
    <div role="img" aria-label={`${axisTitle}: ${rows.map((r) => `${r.label} ${r.valueLabel} ${r.rangeLabel}`).join("; ")}`}>
      <ResponsiveContainer width="100%" height={height}>
        <ScatterChart margin={{ top: 8, right: 24, bottom: 4, left: 0 }}>
          <CartesianGrid horizontal={false} stroke={COLORS.grid} />
          <XAxis
            type="number"
            dataKey="x"
            domain={dom}
            tickFormatter={tickFormat}
            tick={CHART.axisTick}
            axisLine={CHART.axisLine}
            tickLine={false}
            tickMargin={6}
            label={{ value: axisTitle, position: "insideBottom", offset: -4, fontSize: 12, fill: COLORS.axisTitle }}
            height={44}
          />
          <YAxis
            type="number"
            dataKey="y"
            domain={[-0.5, rows.length - 0.5]}
            ticks={rows.map((_, i) => i)}
            reversed
            axisLine={false}
            tickLine={false}
            width={narrow ? 112 : 168}
            tick={(p) => <RowTick {...p} rows={rows} narrow={narrow} />}
            interval={0}
          />
          {refLines.map((l) => (
            <ReferenceLine
              key={l.label}
              x={l.x}
              stroke={l.color ?? COLORS.refSoft}
              strokeDasharray="4 3"
              label={{ value: l.label, position: "insideTopRight", fontSize: 11, fill: COLORS.axis }}
            />
          ))}
          <Tooltip
            cursor={false}
            isAnimationActive={false}
            content={({ active, payload }) => {
              const row = (payload?.[0]?.payload as Pt | undefined)?.row;
              if (!active || !row) return null;
              return (
                <TooltipCard
                  title={row.label}
                  color={row.color}
                  rows={[
                    { label: "Estimate", value: row.valueLabel },
                    { label: "95% interval", value: row.rangeLabel },
                  ]}
                  footnote={row.tooltipNote}
                />
              );
            }}
          />
          {rows.map((row, i) => (
            <Scatter
              key={row.key}
              data={[{ x: row.value, y: i, err: [row.value - row.low, row.high - row.value], row } satisfies Pt]}
              shape={(p: ShapeProps) => <Marker {...p} />}
              isAnimationActive={animate}
              animationDuration={300}
            >
              <ErrorBar dataKey="err" direction="x" width={10} stroke={row.color} strokeWidth={row.emphasis ? 2.25 : 1.75} />
            </Scatter>
          ))}
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
