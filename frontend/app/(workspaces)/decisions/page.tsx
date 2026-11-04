import type { Metadata } from "next";
import { PageHeader } from "@/components/data/page-header";
import { ComingSoon } from "@/components/states/coming-soon";

export const metadata: Metadata = { title: "Decision Studio" };

export default function Page() {
  return (
    <div className="animate-fade-in space-y-6">
      <PageHeader
        overline="04 · Decision Studio"
        title="Decision Studio"
        description="Rank customers under a contact budget."
      />
      <ComingSoon workspace="decisions" milestone="F3" />
    </div>
  );
}
