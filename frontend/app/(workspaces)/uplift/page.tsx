import type { Metadata } from "next";
import { PageHeader } from "@/components/data/page-header";
import { ComingSoon } from "@/components/states/coming-soon";

export const metadata: Metadata = { title: "Uplift Explorer" };

export default function Page() {
  return (
    <div className="animate-fade-in space-y-6">
      <PageHeader
        overline="03 · Uplift Explorer"
        title="Uplift Explorer"
        description="Model leaderboard, scaling and selection."
      />
      <ComingSoon workspace="uplift" milestone="F2" />
    </div>
  );
}
