import type { Metadata } from "next";
import { Suspense } from "react";
import { PageHeader, Section } from "@/components/data/page-header";
import { KpiRowSkeleton, PanelSkeleton } from "@/components/states/skeletons";
import {
  CriteoArms,
  CriteoIpwAipw,
  CriteoLabKpis,
  CriteoPropensity,
  CriteoRandomization,
  HillstromLabSection,
} from "@/features/experiments/sections";
import { WORKSPACES, parseDataset } from "@/lib/datasets";

export const metadata: Metadata = { title: "Experiment Lab" };

const workspace = WORKSPACES.find((w) => w.id === "experiments")!;

export default async function ExperimentLabPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const dataset = parseDataset((await searchParams).dataset, workspace);
  return (
    <div className="animate-fade-in space-y-6 lg:space-y-8">
      <PageHeader
        overline="02 · Experiment Lab"
        title={dataset === "hillstrom" ? "What the e-mail experiment shows" : "Is the Criteo experiment randomized?"}
        description={
          dataset === "hillstrom"
            ? "Treatment effects of the Men's and Women's e-mails against no e-mail, with Holm-adjusted significance, covariate balance and the evidence behind each claim."
            : "Randomization diagnostics, treatment balance and the documented disagreement between IPW and AIPW, all reported before any model is trusted."
        }
        dataset={dataset}
      />

      {dataset === "hillstrom" ? (
        <Suspense key="hs-lab" fallback={<LabSkeleton />}>
          <HillstromLabSection />
        </Suspense>
      ) : (
        <>
          <Suspense key="c-kpi" fallback={<KpiRowSkeleton />}>
            <CriteoLabKpis />
          </Suspense>
          <Suspense key="c-rand" fallback={<PanelSkeleton height={420} />}>
            <CriteoRandomization />
          </Suspense>
          <Suspense key="c-prop" fallback={<PanelSkeleton height={260} />}>
            <CriteoPropensity />
          </Suspense>
          <Section>
            <Suspense key="c-ipw" fallback={<PanelSkeleton height={520} />}>
              <CriteoIpwAipw />
            </Suspense>
          </Section>
          <Suspense key="c-arms" fallback={<PanelSkeleton height={160} />}>
            <CriteoArms />
          </Suspense>
        </>
      )}
    </div>
  );
}

function LabSkeleton() {
  return (
    <div className="space-y-4">
      <PanelSkeleton height={80} />
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
        <PanelSkeleton height={280} className="lg:col-span-7" />
        <PanelSkeleton height={280} className="lg:col-span-5" />
      </div>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
        <PanelSkeleton height={240} className="lg:col-span-8" />
        <PanelSkeleton height={240} className="lg:col-span-4" />
      </div>
    </div>
  );
}
