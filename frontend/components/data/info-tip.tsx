"use client";

import { Info } from "lucide-react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

/** Small "i" that explains a term (AIPW, Holm, Qini...) on hover or keyboard focus. */
export function InfoTip({ children, label = "More information" }: { children: React.ReactNode; label?: string }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          aria-label={label}
          className="inline-flex size-4 items-center justify-center rounded-full text-fg-muted transition-colors hover:text-fg"
        >
          <Info className="size-3.5" />
        </button>
      </TooltipTrigger>
      <TooltipContent>{children}</TooltipContent>
    </Tooltip>
  );
}
