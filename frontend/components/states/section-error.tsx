import { CloudOff, FileQuestion, ServerCrash, Cpu } from "lucide-react";
import { Panel } from "@/components/data/panel";
import type { ApiError } from "@/lib/api/errors";
import { cn } from "@/lib/utils";
import { RetryButton } from "./retry-button";

const COPY: Record<
  ApiError["kind"],
  { title: string; hint: string; icon: typeof CloudOff; retry: boolean }
> = {
  network: {
    title: "The API is not reachable",
    hint: "Check that the UpliftIQ backend is running, then retry. Other sections keep working.",
    icon: CloudOff,
    retry: true,
  },
  report_unavailable: {
    title: "This report isn't available in this deployment",
    hint: "The committed report file this section reads is missing. Other sections are unaffected.",
    icon: FileQuestion,
    retry: false,
  },
  model_unavailable: {
    title: "The scoring model isn't loaded",
    hint: "The model package could not be loaded. Stored results are still available.",
    icon: Cpu,
    retry: true,
  },
  not_found: {
    title: "Not found",
    hint: "The API has no data for this request.",
    icon: FileQuestion,
    retry: false,
  },
  invalid_input: { title: "Invalid input", hint: "", icon: ServerCrash, retry: false },
  invalid_request: { title: "Invalid request", hint: "", icon: ServerCrash, retry: false },
  server: {
    title: "The API returned an error",
    hint: "Something went wrong while reading this section. Retry, or check the backend logs.",
    icon: ServerCrash,
    retry: true,
  },
};

/** Section-level failure: this panel explains itself while the rest of the page keeps working. */
export function SectionError({
  error,
  what,
  className,
  minHeight,
}: {
  error: ApiError;
  /** what the section was going to show, e.g. "Held-out policy values" */
  what: string;
  className?: string;
  minHeight?: number;
}) {
  const c = COPY[error.kind];
  const Icon = c.icon;
  return (
    <Panel
      role="alert"
      className={cn("flex flex-col items-start justify-center gap-3 p-5 sm:p-6", className)}
      style={{ minHeight }}
    >
      <span className="flex size-9 items-center justify-center rounded-control bg-danger-soft text-danger">
        <Icon className="size-[18px]" aria-hidden />
      </span>
      <div>
        <p className="text-sm font-semibold text-fg">{c.title}</p>
        <p className="mt-0.5 text-xs leading-5 text-fg-muted">
          <span className="font-medium text-fg-secondary">{what}.</span> {c.hint}
        </p>
        <p className="mt-2 break-words font-mono text-[11px] leading-4 text-fg-muted">{error.message}</p>
      </div>
      {c.retry ? <RetryButton /> : null}
    </Panel>
  );
}
