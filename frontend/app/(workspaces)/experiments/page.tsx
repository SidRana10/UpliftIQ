import type { Metadata } from "next";
import { PageHeader } from "@/components/data/page-header";
import { ComingSoon } from "@/components/states/coming-soon";

export const metadata: Metadata = { title: "Experiment Lab" };

export default function Page() {
  return (
    <div className="animate-fade-in space-y-6">
      <PageHeader
        overline="02 · Experiment Lab"
        title="Experiment Lab"
        description="Randomized evidence, effects and diagnostics."
      />
      <ComingSoon workspace="experiments" milestone="F2" />
    </div>
  );
}
