import { Badge } from "@/components/ui/badge";

export type EvidenceKind =
  | "heldout"
  | "validation"
  | "cross-validated"
  | "primary"
  | "exploratory"
  | "supported"
  | "stored"
  | "live"
  | "synthetic"
  | "historical"
  | "selected";

const MAP: Record<EvidenceKind, { label: string; tone: React.ComponentProps<typeof Badge>["tone"] }> = {
  heldout: { label: "Held-out", tone: "teal-outline" },
  validation: { label: "Validation", tone: "violet-outline" },
  "cross-validated": { label: "Cross-validated", tone: "violet-outline" },
  primary: { label: "Primary", tone: "teal" },
  exploratory: { label: "Exploratory", tone: "amber" },
  supported: { label: "Supported", tone: "teal" },
  stored: { label: "Stored measurement", tone: "slate" },
  live: { label: "Live", tone: "teal" },
  synthetic: { label: "Synthetic", tone: "amber-solid" },
  historical: { label: "Historical", tone: "slate" },
  selected: { label: "Selected", tone: "teal" },
};

export function EvidenceBadge({ kind, label }: { kind: EvidenceKind; label?: string }) {
  const m = MAP[kind];
  return <Badge tone={m.tone}>{label ?? m.label}</Badge>;
}
