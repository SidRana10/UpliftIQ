import type { Metadata } from "next";
import { Suspense } from "react";
import { PageHeader } from "@/components/data/page-header";
import { KpiRowSkeleton, PanelSkeleton } from "@/components/states/skeletons";
import {
  CaveatSection,
  ChartSection,
  ContinueSection,
  EvidenceSection,
  InsightSection,
  KpiSection,
  OtherStudySection,
  OutcomesSection,
  StatusSection,
} from "@/features/overview/sections";
import { WORKSPACES, parseDataset } from "@/lib/datasets";

export const metadata: Metadata = { title: "Overview" };

const workspace = WORKSPACES.find((w) => w.id === "overview")!;

export default async function OverviewPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const dataset = parseDataset((await searchParams).dataset, workspace);
  // keyed on dataset so switching shows skeletons instead of stale content
  const k = dataset;
  return (
    <div className="animate-fade-in space-y-6 lg:space-y-8">
      <PageHeader
        overline="01 · Overview"
        title="Two marketing studies. Evidence-backed targeting decisions."
        description={
          dataset === "hillstrom"
            ? "A randomized e-mail experiment and its frozen campaign decision, evaluated once on held-out customers."
            : "An anonymized uplift benchmark and its frozen targeting model, evaluated once on a held-out test set."
        }
        dataset={dataset}
      />

      <Suspense key={`kpi-${k}`} fallback={<KpiRowSkeleton />}>
        <KpiSection dataset={dataset} />
      </Suspense>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
        <div className="min-w-0 lg:col-span-8">
          <Suspense key={`chart-${k}`} fallback={<PanelSkeleton height={340} />}>
            <ChartSection dataset={dataset} />
          </Suspense>
        </div>
        <div className="min-w-0 lg:col-span-4">
          <Suspense key={`insight-${k}`} fallback={<PanelSkeleton height={340} />}>
            <InsightSection dataset={dataset} />
          </Suspense>
        </div>
      </div>

      <Suspense key={`caveat-${k}`} fallback={null}>
        <CaveatSection dataset={dataset} />
      </Suspense>

      {dataset === "hillstrom" ? (
        <Suspense key="outcomes" fallback={<PanelSkeleton height={160} />}>
          <OutcomesSection />
        </Suspense>
      ) : null}

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
        <Suspense key={`other-${k}`} fallback={<PanelSkeleton height={160} />}>
          <OtherStudySection dataset={dataset} />
        </Suspense>
        <Suspense key="status" fallback={<PanelSkeleton height={160} />}>
          <StatusSection />
        </Suspense>
        <div className="md:col-span-2 xl:col-span-1">
          <Suspense key={`evidence-${k}`} fallback={<PanelSkeleton height={160} />}>
            <EvidenceSection dataset={dataset} />
          </Suspense>
        </div>
      </div>

      <section aria-labelledby="continue">
        <h2 id="continue" className="mb-3 text-base font-semibold text-fg">
          Continue the analysis
        </h2>
        <ContinueSection dataset={dataset} />
      </section>
    </div>
  );
}
