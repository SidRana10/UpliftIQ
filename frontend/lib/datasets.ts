export type DatasetId = "hillstrom" | "criteo";
export type WorkspaceId = "overview" | "experiments" | "uplift" | "decisions" | "operations";

export const DATASETS: Record<DatasetId, { label: string; kind: string }> = {
  hillstrom: { label: "Hillstrom", kind: "Business email experiment" },
  criteo: { label: "Criteo", kind: "Anonymized benchmark" },
};

export interface Workspace {
  id: WorkspaceId;
  number: string;
  label: string;
  href: string;
  purpose: string;
  /** datasets the workspace can show; `null` = the dataset switch is not offered */
  datasets: DatasetId[] | null;
  defaultDataset: DatasetId;
}

export const WORKSPACES: Workspace[] = [
  {
    id: "overview",
    number: "01",
    label: "Overview",
    href: "/overview",
    purpose: "The headline decision of each study",
    datasets: ["hillstrom", "criteo"],
    defaultDataset: "hillstrom",
  },
  {
    id: "experiments",
    number: "02",
    label: "Experiment Lab",
    href: "/experiments",
    purpose: "Randomized evidence, effects and diagnostics",
    datasets: ["hillstrom", "criteo"],
    defaultDataset: "hillstrom",
  },
  {
    id: "uplift",
    number: "03",
    label: "Uplift Explorer",
    href: "/uplift",
    purpose: "Model leaderboard, scaling and selection",
    datasets: ["hillstrom", "criteo"],
    defaultDataset: "criteo",
  },
  {
    id: "decisions",
    number: "04",
    label: "Decision Studio",
    href: "/decisions",
    purpose: "Rank customers under a contact budget",
    datasets: ["hillstrom", "criteo"],
    defaultDataset: "criteo",
  },
  {
    id: "operations",
    number: "05",
    label: "Model Operations",
    href: "/operations",
    purpose: "Provenance, health and lineage",
    datasets: null,
    defaultDataset: "criteo",
  },
];

export function workspaceFor(pathname: string): Workspace | undefined {
  return WORKSPACES.find((w) => pathname === w.href || pathname.startsWith(`${w.href}/`));
}

export function isDataset(value: unknown): value is DatasetId {
  return value === "hillstrom" || value === "criteo";
}

/** Resolve the `?dataset=` value for a workspace; unsupported or missing values fall back. */
export function parseDataset(raw: string | string[] | undefined, workspace: Workspace): DatasetId {
  const value = Array.isArray(raw) ? raw[0] : raw;
  if (isDataset(value) && (workspace.datasets === null || workspace.datasets.includes(value))) {
    return value;
  }
  return workspace.defaultDataset;
}

/** Link target that carries the current dataset when the destination supports it. */
export function hrefWithDataset(workspace: Workspace, dataset: DatasetId | null): string {
  if (!dataset || !workspace.datasets || !workspace.datasets.includes(dataset)) {
    return workspace.href;
  }
  return `${workspace.href}?dataset=${dataset}`;
}
