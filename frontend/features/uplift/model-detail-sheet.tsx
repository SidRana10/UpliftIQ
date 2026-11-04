"use client";

import { X } from "lucide-react";
import { Dialog } from "radix-ui";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Hash } from "@/components/data/hash";
import type { DetailRow, ModelDetail } from "./select";

function Rows({ title, rows }: { title: string; rows: DetailRow[] }) {
  if (!rows.length) return null;
  return (
    <section>
      <h3 className="eyebrow">{title}</h3>
      <dl className="mt-2 divide-y divide-border/70 rounded-control border border-border">
        {rows.map((r) => (
          <div key={r.label} className="flex justify-between gap-4 px-3 py-2 text-[13px]">
            <dt className="text-fg-muted">{r.label}</dt>
            <dd className="break-all text-right font-medium tabular-nums text-fg">{r.value}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

/** Right-hand model card from the stored manifest: configuration, training, libraries, code commit. */
export function ModelDetailSheet({ detail, onClose }: { detail: ModelDetail | null; onClose: () => void }) {
  return (
    <Dialog.Root open={detail !== null} onOpenChange={(o) => !o && onClose()}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-fg/40 data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:animate-in data-[state=open]:fade-in-0" />
        <Dialog.Content className="fixed inset-y-0 right-0 z-50 flex w-[420px] max-w-[94vw] flex-col bg-surface shadow-popover duration-200 data-[state=closed]:animate-out data-[state=closed]:slide-out-to-right data-[state=open]:animate-in data-[state=open]:slide-in-from-right motion-reduce:animate-none">
          {detail ? (
            <>
              <header className="flex items-start justify-between gap-3 border-b border-border px-5 py-4">
                <div>
                  <Dialog.Title className="text-base font-semibold text-fg">{detail.label}</Dialog.Title>
                  <Dialog.Description className="mt-0.5 text-xs text-fg-muted">
                    Model card · family {detail.family} · config <span className="font-mono">{detail.configKey}</span>
                  </Dialog.Description>
                  <div className="mt-2 flex gap-1.5">
                    <Badge tone="slate">Stored measurement</Badge>
                    <Badge tone="violet-outline">Validation run</Badge>
                  </div>
                </div>
                <Dialog.Close asChild>
                  <Button variant="ghost" size="icon" aria-label="Close model card" className="-mr-2">
                    <X />
                  </Button>
                </Dialog.Close>
              </header>
              <div className="scrollbar-thin flex-1 space-y-5 overflow-y-auto px-5 py-4">
                <Rows title="Run" rows={detail.overview} />
                <Rows title="Hyperparameters" rows={detail.hyperparameters} />
                <Rows title="Training" rows={detail.training} />
                <Rows title="Libraries" rows={detail.libraries} />
                <section>
                  <h3 className="eyebrow">Code</h3>
                  <div className="mt-2 flex flex-wrap items-center gap-2 text-[13px]">
                    <Hash value={detail.commit} />
                    <Badge tone={detail.dirty ? "amber" : "teal"}>{detail.dirty ? "Uncommitted changes when run" : "Clean tree"}</Badge>
                  </div>
                </section>
              </div>
            </>
          ) : null}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
