"use client";

import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  ErrorBar,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { TooltipCard } from "@/components/charts/chart-tooltip";
import { CHART, useReducedMotion } from "@/components/charts/theme";
import { COLORS } from "@/lib/colors";
import { count, ms, seconds, sizeLabel } from "@/lib/format";
import {
  METRIC_LABEL,
  type BudgetRefs,
  type BudgetSeries,
  type LatencyRow,
  type MetricKey,
  type ModelSeries,
  batchLabel,
  fmtEst,
} from "./select";

const pp = (x: number) => x * 100;

// ============================================================================ budget chart

interface BPt {
  x: number;
  y: number;
  err?: [number, number];
  color: string;
  hollow?: boolean;
  title: string;
  rows: { label: string; value: string }[];
  footnote?: string;
}

interface ShapeProps {
  cx?: number;
  cy?: number;
  payload?: BPt;
}

function Marker({ cx = 0, cy = 0, payload }: ShapeProps) {
  if (!payload) return null;
  return (
    <g>
      <circle cx={cx} cy={cy} r={14} fill="transparent" />
      <circle
        cx={cx}
        cy={cy}
        r={payload.hollow ? 4.5 : 5.5}
        fill={payload.hollow ? "#fff" : payload.color}
        stroke={payload.color}
        strokeWidth={payload.hollow ? 2 : 1.5}
      />
    </g>
  );
}

/**
 * Validation policy value at the evaluated budgets (10/20/30%) for the compared finalists: markers
 * with 95% whiskers and no line between budgets. Random-at-budget and treat-all are exact stored
 * point values (the validation report stores their intervals as text only).
 */
export function ModelBudgetChart({
  series,
  refs,
  height = CHART.height.primary,
}: {
  series: BudgetSeries[];
  refs: BudgetRefs;
  height?: number;
}) {
  const animate = !useReducedMotion();
  const budgets = [10, 20, 30];
  const n = series.length;
  const spread = 2.4;
  const top = Math.max(
    refs.treatAll ? pp(refs.treatAll.value) : 0,
    ...series.flatMap((s) => s.points.map((p) => pp(p.est.high))),
  );
  const yMax = Math.ceil(top * 1.1 * 5) / 5;

  const modelData = (s: BudgetSeries, i: number): BPt[] =>
    s.points.map((p) => ({
      x: p.budget + (i - (n - 1) / 2) * spread,
      y: pp(p.est.value),
      err: [pp(p.est.value - p.est.low), pp(p.est.high - p.est.value)],
      color: s.color,
      title: `${s.label}, top ${p.budget}%`,
      rows: [
        { label: "AIPW value", value: `${p.label} pp` },
        { label: "95% interval", value: `${p.range} pp` },
        { label: "Minus random (paired)", value: p.vsRandom },
      ],
      footnote: "Validation estimate (500K subset); the model was selected on these numbers.",
    }));
  const randomData: BPt[] = refs.random.map((r) => ({
    x: r.budget,
    y: pp(r.value),
    color: COLORS.ref,
    hollow: true,
    title: `Random targeting, ${r.budget}%`,
    rows: [
      { label: "AIPW value", value: r.label },
      ...(r.reportText ? [{ label: "Report (95% CI)", value: `${r.reportText} pp` }] : []),
    ],
  }));

  return (
    <div
      role="img"
      aria-label={`Validation policy value at 10, 20 and 30 percent contact budgets for ${series.map((s) => s.label).join(", ")}, with random targeting and treat-all references`}
    >
      <ResponsiveContainer width="100%" height={height}>
        <ScatterChart margin={{ top: 14, right: 20, bottom: 4, left: 0 }}>
          <CartesianGrid vertical={false} stroke={COLORS.grid} />
          <XAxis
            type="number"
            dataKey="x"
            domain={[2, 38]}
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
            tickFormatter={(v: number) => v.toFixed(1)}
            tick={CHART.axisTick}
            axisLine={false}
            tickLine={false}
            width={44}
            label={{ value: "AIPW policy value (pp)", angle: -90, position: "insideLeft", offset: 8, fontSize: 12, fill: COLORS.axisTitle, style: { textAnchor: "middle" } }}
          />
          {refs.treatAll ? (
            <ReferenceLine
              y={pp(refs.treatAll.value)}
              stroke={COLORS.ref}
              strokeDasharray="5 4"
              ifOverflow="visible"
              label={{ value: `Treat everyone ${refs.treatAll.label}`, position: "insideTopLeft", fontSize: 11, fill: COLORS.axisTitle, offset: 6 }}
            />
          ) : null}
          <ReferenceLine y={0} stroke={COLORS.zero} />
          <Tooltip
            cursor={false}
            isAnimationActive={false}
            content={({ active, payload }) => {
              const p = payload?.[0]?.payload as BPt | undefined;
              if (!active || !p) return null;
              return <TooltipCard title={p.title} color={p.color} rows={p.rows} footnote={p.footnote} />;
            }}
          />
          <Scatter data={randomData} shape={(p: ShapeProps) => <Marker {...p} />} isAnimationActive={animate} animationDuration={300} />
          {series.map((s, i) => (
            <Scatter key={s.model} data={modelData(s, i)} shape={(p: ShapeProps) => <Marker {...p} />} isAnimationActive={animate} animationDuration={300}>
              <ErrorBar isAnimationActive={animate} dataKey="err" direction="y" width={8} stroke={s.color} strokeWidth={2} />
            </Scatter>
          ))}
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

// ============================================================================ learning curve

interface Row {
  size: number;
  [k: string]: number | number[] | string | boolean | undefined;
}

function rowsBySize(series: ModelSeries[], metric: MetricKey): Row[] {
  const sizes = [...new Set(series.flatMap((s) => s.points.map((p) => p.size)))].sort((a, b) => a - b);
  return sizes.map((size) => {
    const row: Row = { size };
    for (const s of series) {
      const p = s.points.find((x) => x.size === size);
      if (!p) continue;
      const e = p.est[metric];
      row[`${s.model}_v`] = pp(e.value);
      row[`${s.model}_r`] = [pp(e.low), pp(e.high)];
      row[`${s.model}_t`] = fmtEst(e).text + " pp";
      row[`${s.model}_f`] = p.fullScale;
    }
    return row;
  });
}

function logTicks(sizes: number[]) {
  return sizes;
}

function Diamond({ cx = 0, cy = 0, color }: { cx?: number; cy?: number; color: string }) {
  return <path d={`M${cx},${cy - 7} L${cx + 6},${cy} L${cx},${cy + 7} L${cx - 6},${cy} Z`} fill={color} stroke="#fff" strokeWidth={1.2} />;
}

/**
 * Metric vs training rows on a log axis, with a 95% band per model. Six models are measured at
 * 100K/500K/2M; the finalists have one more point at full scale (filled diamond).
 */
export function LearningCurveChart({
  series,
  metric,
  height = 340,
}: {
  series: ModelSeries[];
  metric: MetricKey;
  height?: number;
}) {
  const animate = !useReducedMotion();
  const data = rowsBySize(series, metric);
  const sizes = data.map((d) => d.size);
  const all = series.flatMap((s) => s.points.map((p) => p.est[metric]));
  const lo = Math.floor(Math.min(...all.map((e) => pp(e.low))) * 10) / 10;
  const hi = Math.ceil(Math.max(...all.map((e) => pp(e.high))) * 10) / 10;
  return (
    <div role="img" aria-label={`${METRIC_LABEL[metric]} against training rows for ${series.map((s) => s.label).join(", ")}`}>
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart data={data} margin={{ top: 12, right: 22, bottom: 4, left: 0 }}>
          <CartesianGrid vertical={false} stroke={COLORS.grid} />
          <XAxis
            type="number"
            dataKey="size"
            scale="log"
            domain={[sizes[0] * 0.75, sizes[sizes.length - 1] * 1.3]}
            ticks={logTicks(sizes)}
            tickFormatter={(v: number) => sizeLabel(v)}
            tick={CHART.axisTick}
            axisLine={CHART.axisLine}
            tickLine={false}
            tickMargin={6}
            height={44}
            label={{ value: "Training rows (log scale)", position: "insideBottom", offset: -2, fontSize: 12, fill: COLORS.axisTitle }}
          />
          <YAxis
            domain={[lo, hi]}
            tickFormatter={(v: number) => v.toFixed(2)}
            tick={CHART.axisTick}
            axisLine={false}
            tickLine={false}
            width={48}
            label={{ value: `${METRIC_LABEL[metric].replace(" (primary)", "")} (pp)`, angle: -90, position: "insideLeft", offset: 8, fontSize: 12, fill: COLORS.axisTitle, style: { textAnchor: "middle" } }}
          />
          <Tooltip
            cursor={{ stroke: COLORS.zero }}
            isAnimationActive={false}
            content={({ active, payload }) => {
              const row = payload?.[0]?.payload as Row | undefined;
              if (!active || !row) return null;
              const items = series.filter((s) => row[`${s.model}_t`] !== undefined);
              return (
                <TooltipCard
                  title={`${count(row.size)} training rows`}
                  rows={items.map((s) => ({ label: s.label, value: String(row[`${s.model}_t`]) }))}
                  footnote="Validation estimates with 95% intervals."
                />
              );
            }}
          />
          {series.map((s) => (
            <Area
              key={`${s.model}-band`}
              dataKey={`${s.model}_r`}
              stroke="none"
              fill={s.color}
              fillOpacity={0.1}
              isAnimationActive={animate}
              animationDuration={300}
              legendType="none"
              activeDot={false}
            />
          ))}
          {series.map((s) => (
            <Line
              key={s.model}
              dataKey={`${s.model}_v`}
              stroke={s.color}
              strokeWidth={s.model === "s_learner" ? 2.5 : 2}
              isAnimationActive={animate}
              animationDuration={300}
              dot={(p: { cx?: number; cy?: number; payload?: Row; index?: number }) => {
                const full = p.payload?.[`${s.model}_f`] === true;
                return full ? (
                  <Diamond key={`${s.model}-${p.index}`} cx={p.cx} cy={p.cy} color={s.color} />
                ) : (
                  <circle key={`${s.model}-${p.index}`} cx={p.cx} cy={p.cy} r={3.8} fill="#fff" stroke={s.color} strokeWidth={2} />
                );
              }}
              activeDot={{ r: 5 }}
            />
          ))}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

// ============================================================================ training time

function decadeAxis(values: number[]) {
  const lo = 10 ** Math.floor(Math.log10(Math.min(...values) / 1.2));
  const hi = 10 ** Math.ceil(Math.log10(Math.max(...values) * 1.2));
  const ticks: number[] = [];
  for (let v = lo; v <= hi * 1.001; v *= 10) ticks.push(v);
  return { domain: [lo, hi] as [number, number], ticks };
}

/** Training seconds vs training rows, log-log, from the stored engineering measurements. */
export function TrainingTimeChart({ series, height = CHART.height.secondary + 20 }: { series: ModelSeries[]; height?: number }) {
  const animate = !useReducedMotion();
  const sizes = [...new Set(series.flatMap((s) => s.points.map((p) => p.size)))].sort((a, b) => a - b);
  const data = sizes.map((size) => {
    const row: Row = { size };
    for (const s of series) {
      const p = s.points.find((x) => x.size === size);
      if (p && p.trainSeconds !== null) {
        row[`${s.model}_v`] = p.trainSeconds;
        row[`${s.model}_t`] = seconds(p.trainSeconds);
        row[`${s.model}_f`] = p.fullScale;
      }
    }
    return row;
  });
  const vals = series.flatMap((s) => s.points.flatMap((p) => (p.trainSeconds !== null ? [p.trainSeconds] : [])));
  const y = decadeAxis(vals);
  return (
    <div role="img" aria-label={`Training time against training rows for ${series.map((s) => s.label).join(", ")}`}>
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart data={data} margin={{ top: 10, right: 18, bottom: 4, left: 0 }}>
          <CartesianGrid vertical={false} stroke={COLORS.grid} />
          <XAxis type="number" dataKey="size" scale="log" domain={[sizes[0] * 0.75, sizes[sizes.length - 1] * 1.3]} ticks={sizes} tickFormatter={(v: number) => sizeLabel(v)} tick={CHART.axisTick} axisLine={CHART.axisLine} tickLine={false} tickMargin={6} height={40} label={{ value: "Training rows (log)", position: "insideBottom", offset: -2, fontSize: 12, fill: COLORS.axisTitle }} />
          <YAxis scale="log" domain={y.domain} ticks={y.ticks} allowDataOverflow tickFormatter={(v: number) => (v >= 1 ? `${v}` : `${v}`)} tick={CHART.axisTick} axisLine={false} tickLine={false} width={44} label={{ value: "Seconds (log)", angle: -90, position: "insideLeft", offset: 10, fontSize: 12, fill: COLORS.axisTitle, style: { textAnchor: "middle" } }} />
          <Tooltip
            cursor={{ stroke: COLORS.zero }}
            isAnimationActive={false}
            content={({ active, payload }) => {
              const row = payload?.[0]?.payload as Row | undefined;
              if (!active || !row) return null;
              return (
                <TooltipCard
                  title={`${count(row.size)} training rows`}
                  rows={series.filter((s) => row[`${s.model}_t`] !== undefined).map((s) => ({ label: s.label, value: String(row[`${s.model}_t`]) }))}
                  footnote="Stored offline measurement, one machine, OMP_NUM_THREADS=1."
                />
              );
            }}
          />
          {series.map((s) => (
            <Line
              key={s.model}
              dataKey={`${s.model}_v`}
              stroke={s.color}
              strokeWidth={2}
              isAnimationActive={animate}
              animationDuration={300}
              dot={(p: { cx?: number; cy?: number; payload?: Row; index?: number }) =>
                p.payload?.[`${s.model}_f`] === true ? (
                  <Diamond key={`${s.model}-${p.index}`} cx={p.cx} cy={p.cy} color={s.color} />
                ) : (
                  <circle key={`${s.model}-${p.index}`} cx={p.cx} cy={p.cy} r={3.2} fill="#fff" stroke={s.color} strokeWidth={1.8} />
                )
              }
              activeDot={{ r: 4.5 }}
            />
          ))}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

// ============================================================================ cost vs quality

interface CQ {
  x: number;
  y: number;
  err: [number, number];
  color: string;
  full: boolean;
  label: string;
  title: string;
  rows: { label: string; value: string }[];
}

function CqShape({ cx = 0, cy = 0, payload }: { cx?: number; cy?: number; payload?: CQ }) {
  if (!payload) return null;
  return (
    <g>
      <circle cx={cx} cy={cy} r={14} fill="transparent" />
      {payload.full ? (
        <Diamond cx={cx} cy={cy} color={payload.color} />
      ) : (
        <circle cx={cx} cy={cy} r={4.5} fill="#fff" stroke={payload.color} strokeWidth={2} />
      )}
    </g>
  );
}

/** Training seconds vs the selected quality metric (with 95% whisker): 2M rows for every model, full scale for finalists. */
export function CostQualityChart({ series, metric, height = CHART.height.secondary + 20 }: { series: ModelSeries[]; metric: MetricKey; height?: number }) {
  const animate = !useReducedMotion();
  const per = series.map((s) => {
    const pts = s.points.filter((p) => p.trainSeconds !== null && (p.fullScale || p.size === 2_000_000));
    return {
      s,
      data: pts.map(
        (p): CQ => ({
          x: p.trainSeconds as number,
          y: pp(p.est[metric].value),
          err: [pp(p.est[metric].value - p.est[metric].low), pp(p.est[metric].high - p.est[metric].value)],
          color: s.color,
          full: p.fullScale,
          label: s.label,
          title: `${s.label}, ${sizeLabel(p.size)} rows`,
          rows: [
            { label: METRIC_LABEL[metric].replace(" (primary)", ""), value: `${fmtEst(p.est[metric]).text} pp` },
            { label: "Training time", value: seconds(p.trainSeconds as number) },
          ],
        }),
      ),
    };
  });
  const xs = per.flatMap((p) => p.data.map((d) => d.x));
  const ys = per.flatMap((p) => p.data.flatMap((d) => [d.y - d.err[0], d.y + d.err[1]]));
  const x = decadeAxis(xs);
  const yLo = Math.floor(Math.min(...ys) * 10) / 10;
  const yHi = Math.ceil(Math.max(...ys) * 10) / 10;
  return (
    <div role="img" aria-label={`Training time against ${METRIC_LABEL[metric]} for ${series.map((s) => s.label).join(", ")}`}>
      <ResponsiveContainer width="100%" height={height}>
        <ScatterChart margin={{ top: 12, right: 40, bottom: 4, left: 0 }}>
          <CartesianGrid vertical={false} stroke={COLORS.grid} />
          <XAxis type="number" dataKey="x" scale="log" domain={x.domain} ticks={x.ticks} tick={CHART.axisTick} axisLine={CHART.axisLine} tickLine={false} tickMargin={6} height={40} label={{ value: "Training seconds (log)", position: "insideBottom", offset: -2, fontSize: 12, fill: COLORS.axisTitle }} />
          <YAxis type="number" dataKey="y" domain={[yLo, yHi]} tickFormatter={(v: number) => v.toFixed(2)} tick={CHART.axisTick} axisLine={false} tickLine={false} width={46} label={{ value: `${METRIC_LABEL[metric].replace(" (primary)", "")} (pp)`, angle: -90, position: "insideLeft", offset: 10, fontSize: 12, fill: COLORS.axisTitle, style: { textAnchor: "middle" } }} />
          <Tooltip
            cursor={false}
            isAnimationActive={false}
            content={({ active, payload }) => {
              const p = payload?.[0]?.payload as CQ | undefined;
              if (!active || !p) return null;
              return <TooltipCard title={p.title} color={p.color} rows={p.rows} footnote="Validation estimate; stored offline timing." />;
            }}
          />
          {per.map(({ s, data }) => (
            <Scatter key={s.model} data={data} shape={(p: { cx?: number; cy?: number; payload?: CQ }) => <CqShape {...p} />} isAnimationActive={animate} animationDuration={300}>
              <ErrorBar isAnimationActive={animate} dataKey="err" direction="y" width={6} stroke={s.color} strokeWidth={1.5} />
            </Scatter>
          ))}
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

// ============================================================================ latency

/** Median latency per batch size (bars, log scale) with the 95th percentile as the upper whisker. */
export function LatencyChart({ rows, height = CHART.height.secondary + 20 }: { rows: LatencyRow[]; height?: number }) {
  const animate = !useReducedMotion();
  const batches = rows[0]?.batches.map((b) => b.batch) ?? [];
  const data = batches.map((batch) => {
    const row: Record<string, number | string | number[]> = { batch: batchLabel(batch) };
    for (const r of rows) {
      const b = r.batches.find((x) => x.batch === batch);
      if (b) {
        row[r.model] = b.p50;
        row[`${r.model}_e`] = [0, Math.max(0, b.p95 - b.p50)];
        row[`${r.model}_t`] = `${ms(b.p50)} / ${ms(b.p95)} · ${count(b.rowsPerS)} rows/s`;
      }
    }
    return row;
  });
  const vals = rows.flatMap((r) => r.batches.flatMap((b) => [b.p50, b.p95]));
  const y = decadeAxis(vals);
  return (
    <div role="img" aria-label={`Median prediction latency by batch size for ${rows.map((r) => r.label).join(", ")}`}>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} margin={{ top: 10, right: 12, bottom: 4, left: 0 }} barCategoryGap="22%">
          <CartesianGrid vertical={false} stroke={COLORS.grid} />
          <XAxis dataKey="batch" tick={CHART.axisTick} axisLine={CHART.axisLine} tickLine={false} tickMargin={6} />
          <YAxis scale="log" domain={y.domain} ticks={y.ticks} allowDataOverflow tick={CHART.axisTick} axisLine={false} tickLine={false} width={44} tickFormatter={(v: number) => `${v}`} label={{ value: "Milliseconds (log)", angle: -90, position: "insideLeft", offset: 10, fontSize: 12, fill: COLORS.axisTitle, style: { textAnchor: "middle" } }} />
          <Tooltip
            cursor={{ fill: COLORS.grid, opacity: 0.5 }}
            isAnimationActive={false}
            content={({ active, payload, label }) => {
              if (!active || !payload?.length) return null;
              const row = payload[0].payload as Record<string, string>;
              return (
                <TooltipCard
                  title={`Batch of ${String(label)}`}
                  rows={rows.filter((r) => row[`${r.model}_t`]).map((r) => ({ label: r.label, value: row[`${r.model}_t`] }))}
                  footnote="p50 / p95 latency · throughput. Stored offline measurement."
                />
              );
            }}
          />
          {rows.map((r) => (
            <Bar key={r.model} dataKey={r.model} fill={r.color} radius={[3, 3, 0, 0]} isAnimationActive={animate} animationDuration={300}>
              <ErrorBar isAnimationActive={animate} dataKey={`${r.model}_e`} direction="y" width={5} stroke={COLORS.text} strokeWidth={1.2} />
            </Bar>
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
