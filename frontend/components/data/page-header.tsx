import { FlaskConical, Mail } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { DATASETS, type DatasetId } from "@/lib/datasets";

export function DatasetChip({ dataset }: { dataset: DatasetId }) {
  const Icon = dataset === "hillstrom" ? Mail : FlaskConical;
  return (
    <Badge tone="slate" className="gap-1.5 px-2 py-1 text-xs">
      <Icon aria-hidden />
      <span className="font-semibold text-fg">{DATASETS[dataset].label}</span>
      <span className="text-fg-muted">{DATASETS[dataset].kind}</span>
    </Badge>
  );
}

export function PageHeader({
  overline,
  title,
  description,
  dataset,
  actions,
}: {
  overline: string;
  title: string;
  description: React.ReactNode;
  dataset?: DatasetId;
  actions?: React.ReactNode;
}) {
  return (
    <header className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
      <div className="min-w-0 max-w-3xl">
        <div className="eyebrow">{overline}</div>
        <h1 className="mt-1.5 text-2xl font-semibold leading-8 tracking-[-0.01em] text-fg sm:text-[28px] sm:leading-9">
          {title}
        </h1>
        <p className="mt-2 text-sm leading-6 text-fg-secondary">{description}</p>
      </div>
      <div className="flex shrink-0 flex-wrap items-center gap-2">
        {dataset ? <DatasetChip dataset={dataset} /> : null}
        {actions}
      </div>
    </header>
  );
}

export function Section({
  title,
  description,
  children,
  className,
}: {
  title?: string;
  description?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section className={className}>
      {title ? (
        <div className="mb-3">
          <h2 className="text-base font-semibold leading-6 text-fg">{title}</h2>
          {description ? <p className="mt-0.5 text-[13px] leading-5 text-fg-muted">{description}</p> : null}
        </div>
      ) : null}
      {children}
    </section>
  );
}
