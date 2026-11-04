import type { Metadata } from "next";
import { PageHeader } from "@/components/data/page-header";
import { ComingSoon } from "@/components/states/coming-soon";

export const metadata: Metadata = { title: "Model Operations" };

export default function Page() {
  return (
    <div className="animate-fade-in space-y-6">
      <PageHeader
        overline="05 · Model Operations"
        title="Model Operations"
        description="Provenance, health and lineage."
      />
      <ComingSoon workspace="operations" milestone="F3" />
    </div>
  );
}
