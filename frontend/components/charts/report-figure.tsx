"use client";

import { Maximize2 } from "lucide-react";
import { Dialog } from "radix-ui";
import { Panel, PanelBody, PanelHeader } from "@/components/data/panel";
import { SourceFootnote } from "@/components/data/source-footnote";
import { Button } from "@/components/ui/button";

/**
 * A figure stored with the committed reports (PNG). Used only where the report holds no numeric
 * series for the picture (e.g. the held-out gain curve), so the chart is shown exactly as stored.
 */
export function ReportFigure({
  title,
  subtitle,
  src,
  alt,
  badge,
  source,
  className,
}: {
  title: string;
  subtitle?: React.ReactNode;
  /** same-origin API path, e.g. /api/v1/criteo/figures/heldout_gain_curve */
  src: string;
  alt: string;
  badge?: React.ReactNode;
  source: string;
  className?: string;
}) {
  return (
    <Panel className={className}>
      <PanelHeader title={title} subtitle={subtitle} actions={badge} />
      <PanelBody>
        <Dialog.Root>
          <Dialog.Trigger asChild>
            <button
              type="button"
              className="group relative block w-full cursor-zoom-in overflow-hidden rounded-control border border-border bg-white"
              aria-label={`Enlarge figure: ${title}`}
            >
              {/* eslint-disable-next-line @next/next/no-img-element -- stored report PNG served by the API proxy */}
              <img src={src} alt={alt} loading="lazy" className="block h-auto w-full" />
              <span className="absolute right-2 top-2 rounded-chip bg-surface/90 p-1 text-fg-muted opacity-0 shadow-panel transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100">
                <Maximize2 className="size-3.5" aria-hidden />
              </span>
            </button>
          </Dialog.Trigger>
          <Dialog.Portal>
            <Dialog.Overlay className="fixed inset-0 z-50 bg-fg/50" />
            <Dialog.Content className="fixed left-1/2 top-1/2 z-50 max-h-[92dvh] w-[min(1100px,94vw)] -translate-x-1/2 -translate-y-1/2 overflow-auto rounded-panel bg-surface p-4 shadow-popover">
              <Dialog.Title className="text-sm font-semibold text-fg">{title}</Dialog.Title>
              <Dialog.Description className="sr-only">{alt}</Dialog.Description>
              {/* eslint-disable-next-line @next/next/no-img-element -- stored report PNG */}
              <img src={src} alt={alt} className="mt-3 block h-auto w-full" />
              <Dialog.Close asChild>
                <Button variant="secondary" size="sm" className="mt-3">
                  Close
                </Button>
              </Dialog.Close>
            </Dialog.Content>
          </Dialog.Portal>
        </Dialog.Root>
        <SourceFootnote sources={[source]} note="Stored report figure (static image)." className="mt-3" />
      </PanelBody>
    </Panel>
  );
}
