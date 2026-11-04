"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  ComposedChart,
  Line,
} from "recharts";
import { CHART, useReducedMotion } from "@/components/charts/theme";
import { TooltipCard } from "@/components/charts/chart-tooltip";
import { COLORS } from "@/lib/colors";
import { count } from "@/lib/format";
import type { ArmBar, StratumPoint } from "./select";

/** Observed outcome by arm (unadjusted). Intervals for the differences are in the forest plot. */
export function ArmBarsChart({
  bars,
  tick,
  axisTitle,
  describe,
}: {
  bars: ArmBar[];
  tick: (v: number) => string;
  axisTitle: string;
  describe: string;
}) {
  const animate = !useReducedMotion();
  const top = Math.max(...bars.map((b) => b.value));
  return (
    <div role="img" aria-label={describe}>
      <ResponsiveContainer width="100%" height={CHART.height.secondary + 40}>
        <BarChart data={bars} margin={{ top: 26, right: 16, bottom: 4, left: 0 }} barCategoryGap="28%">
          <CartesianGrid vertical={false} stroke={COLORS.grid} />
          <XAxis dataKey="label" tick={{ fontSize: 12, fill: COLORS.text }} axisLine={CHART.axisLine} tickLine={false} tickMargin={8} />
          <YAxis
            domain={[0, Math.ceil(top * 1.18 * 10) / 10]}
            tickFormatter={tick}
            tick={CHART.axisTick}
            axisLine={false}
            tickLine={false}
            width={62}
            label={{ value: axisTitle, angle: -90, position: "insideLeft", offset: 8, fontSize: 12, fill: COLORS.axisTitle, style: { textAnchor: "middle" } }}
          />
          <Tooltip
            cursor={{ fill: COLORS.grid, opacity: 0.5 }}
            isAnimationActive={false}
            content={({ active, payload }) => {
              const b = payload?.[0]?.payload as ArmBar | undefined;
              if (!active || !b) return null;
              return (
                <TooltipCard
                  title={b.label}
                  color={b.color}
                  rows={[
                    { label: "Observed", value: b.valueLabel },
                    { label: "Customers", value: count(b.n) },
                  ]}
                  footnote="Observed arm mean, unadjusted. Intervals for the differences are in the forest plot."
                />
              );
            }}
          />
          <ReferenceLine y={0} stroke={COLORS.zero} />
          <Bar dataKey="value" radius={[4, 4, 0, 0]} maxBarSize={88} isAnimationActive={animate} animationDuration={300}>
            {bars.map((b) => (
              <Cell key={b.key} fill={b.color} fillOpacity={b.key === "control" ? 0.55 : 0.9} />
            ))}
            <LabelList dataKey="valueLabel" position="top" fontSize={12} fontWeight={600} fill={COLORS.text} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/**
 * Effect by decile of predicted baseline (top) and treated share by decile (bottom), on one
 * shared category axis. The treated share rising with the baseline is why IPW and AIPW disagree.
 */
export function StrataChart({ points }: { points: StratumPoint[] }) {
  const animate = !useReducedMotion();
  const shareMin = Math.floor(Math.min(...points.map((p) => p.treatedShare)) * 2) / 2 - 0.5;
  const shareMax = Math.ceil(Math.max(...points.map((p) => p.treatedShare)) * 2) / 2 + 0.5;
  return (
    <div
      role="img"
      aria-label={`Treatment effect and treated share across ${points.length} deciles of predicted baseline visit probability`}
      className="space-y-1"
    >
      <div>
        <div className="mb-1 ml-12 text-xs font-medium text-fg-secondary">Observed effect (pp of visit rate)</div>
        <ResponsiveContainer width="100%" height={170}>
          <BarChart data={points} margin={{ top: 14, right: 12, bottom: 0, left: 0 }}>
            <CartesianGrid vertical={false} stroke={COLORS.grid} />
            <XAxis dataKey="stratum" hide />
            <YAxis tick={CHART.axisTick} axisLine={false} tickLine={false} width={44} tickFormatter={(v: number) => v.toFixed(0)} />
            <Tooltip
              cursor={{ fill: COLORS.grid, opacity: 0.5 }}
              isAnimationActive={false}
              content={({ active, payload }) => {
                const p = payload?.[0]?.payload as StratumPoint | undefined;
                if (!active || !p) return null;
                return (
                  <TooltipCard
                    title={`Decile ${p.stratum} of predicted baseline`}
                    color={COLORS.amber}
                    rows={[
                      { label: "Effect", value: p.effectLabel },
                      { label: "Treated share", value: p.treatedShareLabel },
                      { label: "Mean baseline", value: p.baselineLabel },
                      { label: "Visit rates", value: p.rates },
                      { label: "Customers", value: count(p.n) },
                    ]}
                    footnote="Observed within the 500K validation sample; no interval is stored per decile."
                  />
                );
              }}
            />
            <ReferenceLine y={0} stroke={COLORS.zero} />
            <Bar dataKey="effect" fill={COLORS.amber} radius={[3, 3, 0, 0]} isAnimationActive={animate} animationDuration={300} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div>
        <div className="mb-1 ml-12 text-xs font-medium text-fg-secondary">Treated share (%) · design value 85%</div>
        <ResponsiveContainer width="100%" height={150}>
          <ComposedChart data={points} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
            <CartesianGrid vertical={false} stroke={COLORS.grid} />
            <XAxis
              dataKey="stratum"
              tick={CHART.axisTick}
              axisLine={CHART.axisLine}
              tickLine={false}
              tickMargin={6}
              label={{ value: "Decile of predicted baseline visit probability (1 = lowest, 10 = highest)", position: "insideBottom", offset: -4, fontSize: 12, fill: COLORS.axisTitle }}
              height={42}
            />
            <YAxis
              domain={[shareMin, shareMax]}
              tick={CHART.axisTick}
              axisLine={false}
              tickLine={false}
              width={44}
              tickFormatter={(v: number) => v.toFixed(1)}
              ticks={[shareMin + 0.5, 85, shareMax - 0.5].filter((v, i, a) => a.indexOf(v) === i)}
            />
            <ReferenceLine y={85} stroke={COLORS.refSoft} strokeDasharray="4 3" />
            <Tooltip content={() => null} cursor={false} isAnimationActive={false} />
            <Line type="linear" dataKey="treatedShare" stroke={COLORS.teal} strokeWidth={2} dot={{ r: 3.5, fill: COLORS.teal }} isAnimationActive={animate} animationDuration={300} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
