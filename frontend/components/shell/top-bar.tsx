"use client";

import { FlaskConical, Mail, Menu } from "lucide-react";
import { usePathname } from "next/navigation";
import { EvidenceBadge, type EvidenceKind } from "@/components/data/evidence-badge";
import { Button } from "@/components/ui/button";
import { SegmentedControl } from "@/components/ui/segmented-control";
import { DATASETS, WORKSPACES, workspaceFor, type DatasetId, type WorkspaceId } from "@/lib/datasets";
import { useQueryState } from "@/lib/url-state";

const EVIDENCE: Record<WorkspaceId, { kind: EvidenceKind; label: string }> = {
  overview: { kind: "heldout", label: "Held-out, one-shot" },
  experiments: { kind: "primary", label: "Randomized evidence" },
  uplift: { kind: "validation", label: "Validation & held-out" },
  decisions: { kind: "live", label: "Live model output" },
  operations: { kind: "stored", label: "Provenance" },
};

const OPTIONS = (["hillstrom", "criteo"] as DatasetId[]).map((d) => ({
  value: d,
  label: DATASETS[d].label,
  icon: d === "hillstrom" ? <Mail /> : <FlaskConical />,
}));

function DatasetSwitch({ allowed, fallback }: { allowed: DatasetId[]; fallback: DatasetId }) {
  const [dataset, setDataset] = useQueryState<DatasetId>("dataset", fallback, allowed, true);
  return (
    <SegmentedControl<DatasetId>
      label="Dataset"
      value={dataset}
      onValueChange={setDataset}
      options={OPTIONS.filter((o) => allowed.includes(o.value))}
    />
  );
}

export function TopBar({ onOpenNav, dataset }: { onOpenNav: () => void; dataset: DatasetId | null }) {
  const pathname = usePathname();
  const w = workspaceFor(pathname) ?? WORKSPACES[0];
  const ev = EVIDENCE[w.id];
  return (
    <div className="sticky top-0 z-30 border-b border-border bg-surface/90 backdrop-blur supports-[backdrop-filter]:bg-surface/80">
      <div className="flex h-14 items-center gap-3 px-4 sm:px-6 xl:px-8">
        <Button variant="ghost" size="icon" className="-ml-2 md:hidden" onClick={onOpenNav} aria-label="Open navigation">
          <Menu />
        </Button>

        <nav aria-label="Breadcrumb" className="min-w-0 flex-1">
          <ol className="flex min-w-0 items-center gap-1.5 text-[13px] text-fg-muted">
            <li className="hidden shrink-0 sm:block">UpliftIQ</li>
            <li aria-hidden className="hidden text-border-strong sm:block">/</li>
            <li className="truncate font-medium text-fg" aria-current={dataset ? undefined : "page"}>
              {w.label}
            </li>
            {dataset && w.datasets ? (
              <>
                <li aria-hidden className="hidden text-border-strong sm:block">/</li>
                <li className="hidden truncate sm:block" aria-current="page">
                  {DATASETS[dataset].label}
                </li>
              </>
            ) : null}
          </ol>
        </nav>

        <div className="hidden lg:block">
          <EvidenceBadge kind={ev.kind} label={ev.label} />
        </div>
        {w.datasets ? <DatasetSwitch allowed={w.datasets} fallback={w.defaultDataset} /> : null}
      </div>
    </div>
  );
}
