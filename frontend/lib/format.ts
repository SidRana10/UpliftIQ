/**
 * The only place numbers are formatted. Dataset-specific formatters are kept separate so a
 * Criteo value (percentage points of visit rate) can never be passed to a currency formatter.
 */

const MINUS = "−";

function sign(n: number, dp: number): string {
  // avoid "+0.00" / "−0.00" noise: sign follows the rounded value
  const rounded = Number(n.toFixed(dp));
  if (rounded === 0) return "";
  return rounded > 0 ? "+" : MINUS;
}

function fixed(n: number, dp: number): string {
  const s = Math.abs(n).toFixed(dp);
  const neg = Number(n.toFixed(dp)) < 0;
  return neg ? `${MINUS}${s}` : s;
}

// ------------------------------------------------------------------ Hillstrom (business units)

export const hillstrom = {
  /** proportion -> percent number, for chart axes */
  toPercent: (x: number) => x * 100,
  /** proportion -> percent, e.g. 0.1819 -> "18.2%" */
  rate: (x: number, dp = 1) => `${fixed(x * 100, dp)}%`,
  /** proportion difference -> signed percentage points, e.g. 0.0294 -> "+2.94 pp" */
  rateDelta: (x: number, dp = 2) => `${sign(x * 100, dp)}${Math.abs(x * 100).toFixed(dp)} pp`,
  /** dollars per customer */
  spend: (x: number, dp = 2) => `${x < 0 ? MINUS : ""}$${Math.abs(x).toFixed(dp)}`,
  spendDelta: (x: number, dp = 2) => `${sign(x, dp)}$${Math.abs(x).toFixed(dp)}`,
};

// ------------------------------------------------------------------ Criteo (percentage points)

export const criteo = {
  /** stored fraction of visit rate -> percentage-point number, for chart axes */
  toPp: (x: number) => x * 100,
  /** stored fraction of visit rate -> percentage points, e.g. 0.00674 -> "0.674 pp" */
  pp: (x: number, dp = 3) => `${fixed(x * 100, dp)} pp`,
  /** same, without the unit (for use inside an interval) */
  ppValue: (x: number, dp = 3) => fixed(x * 100, dp),
  ppDelta: (x: number, dp = 3) => `${sign(x * 100, dp)}${Math.abs(x * 100).toFixed(dp)} pp`,
  /** predicted probability -> percent */
  prob: (x: number, dp = 2) => `${fixed(x * 100, dp)}%`,
};

// ------------------------------------------------------------------ generic

/** "18.2% [17.1, 19.3]"-style interval; the formatter is applied to all three numbers. */
export function interval(
  value: number,
  low: number,
  high: number,
  f: (n: number) => string,
): { value: string; range: string } {
  return { value: f(value), range: `[${f(low)}, ${f(high)}]` };
}

export function pValue(p: number | null | undefined, label = "p"): string {
  if (p === null || p === undefined || Number.isNaN(p)) return "n/a";
  if (p < 0.001) return `${label} < 0.001`;
  return `${label} = ${p.toFixed(3)}`;
}

export function count(n: number): string {
  return Math.round(n).toLocaleString("en-US");
}

export function compact(n: number): string {
  const abs = Math.abs(n);
  if (abs >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (abs >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return `${Math.round(n)}`;
}

export function ms(x: number): string {
  if (x < 1) return `${x.toFixed(2)} ms`;
  return `${x.toFixed(1)} ms`;
}

export function seconds(x: number): string {
  return `${x.toFixed(1)} s`;
}

export function gigabytes(x: number): string {
  return `${x.toFixed(2)} GB`;
}

export function budget(x: number): string {
  return `${Math.round(x * 100)}%`;
}

/** first 8 and last 6 characters of a hash, e.g. "1e5059cb…65dcf980" style */
export function shortHash(h: string, head = 8, tail = 6): string {
  if (h.length <= head + tail + 1) return h;
  return `${h.slice(0, head)}…${h.slice(-tail)}`;
}

export function dateTime(iso: string | undefined | null): string {
  if (!iso) return "n/a";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    timeZone: "UTC",
    timeZoneName: "short",
  });
}
