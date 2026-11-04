import { Hourglass } from "lucide-react";
import { Panel } from "@/components/data/panel";
import { WORKSPACES, type WorkspaceId } from "@/lib/datasets";

export function ComingSoon({ workspace, milestone }: { workspace: WorkspaceId; milestone: "F2" | "F3" }) {
  const w = WORKSPACES.find((x) => x.id === workspace)!;
  return (
    <Panel className="flex flex-col items-start gap-4 p-6 sm:p-8">
      <span className="flex size-10 items-center justify-center rounded-control bg-violet-soft text-violet-ink">
        <Hourglass className="size-5" aria-hidden />
      </span>
      <div className="max-w-xl">
        <h2 className="text-base font-semibold text-fg">{w.label} arrives in milestone {milestone}</h2>
        <p className="mt-1.5 text-sm leading-6 text-fg-secondary">
          {w.purpose}. This workspace is part of the approved plan and will read the same validated
          reports and live API endpoints as the Overview. The Overview, navigation and design system
          are complete.
        </p>
      </div>
    </Panel>
  );
}
