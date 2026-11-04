/** Literal colours for chart libraries; must mirror the tokens in app/globals.css. */
export const COLORS = {
  teal: "#117E74",
  tealSoft: "#E3F2F0",
  violet: "#7467C9",
  violetSoft: "#ECEAF8",
  amber: "#D99A43",
  amberSoft: "#FBF1E2",
  ref: "#64748B",
  refSoft: "#94A3B8",
  grid: "#E7EBF0",
  axis: "#6B7686",
  axisTitle: "#4A5568",
  text: "#0F1726",
  zero: "#CBD3DD",
  danger: "#C2413B",
} as const;

/** Fixed series colours so a model or policy is the same colour on every chart. */
export const SERIES_COLORS: Record<string, string> = {
  s_learner: "#117E74",
  tarnet_cpu: "#7467C9",
  tarnet_mps: "#A99FE0",
  logistic: "#3E7CB1",
  t_learner: "#B8607A",
  dr_learner: "#7F8F4E",
  mens_for_all: "#117E74",
  womens_for_all: "#7467C9",
  no_email: "#64748B",
  random: "#94A3B8",
};
