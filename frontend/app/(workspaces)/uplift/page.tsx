import type { Metadata } from "next";
import { Suspense } from "react";
import { PageHeader } from "@/components/data/page-header";
import { PanelSkeleton } from "@/components/states/skeletons";
import { CriteoExplorerSection, HillstromExplorerSection } from "@/features/uplift/sections";
import { WORKSPACES, parseDataset } from "@/lib/datasets";

export const metadata: Metadata = { title: "Uplift Explorer" };

const workspace = WORKSPACES.find((w) => w.id === "uplift")!;

export default async function UpliftExplorerPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const dataset = parseDataset((await searchParams).dataset, workspace);
  return (
    <div className="animate-fade-in space-y-6 lg:space-y-8">
      <PageHeader
        overline="03 · Uplift Explorer"
        title={dataset === "criteo" ? "Which uplift model should rank customers?" : "Does personalization beat one e-mail for everyone?"}
        description={
          dataset === "criteo"
            ? "Compare the Criteo uplift models on validation evidence, see how they scale and what they cost, and see why the S-learner was frozen before its one-shot held-out evaluation."
            : "Cross-validated comparison of four personalized models and four fixed policies, and the rule that chose the final Hillstrom policy."
        }
        dataset={dataset}
      />
      {dataset === "criteo" ? (
        <Suspense key="criteo-explorer" fallback={<ExplorerSkeleton />}>
          <CriteoExplorerSection />
        </Suspense>
      ) : (
        <Suspense key="hs-explorer" fallback={<ExplorerSkeleton />}>
          <HillstromExplorerSection />
        </Suspense>
      )}
    </div>
  );
}

function ExplorerSkeleton() {
  return (
    <div className="space-y-4">
      <PanelSkeleton height={60} />
      <PanelSkeleton height={220} />
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
        <PanelSkeleton height={300} className="lg:col-span-7" />
        <PanelSkeleton height={300} className="lg:col-span-5" />
      </div>
    </div>
  );
}
