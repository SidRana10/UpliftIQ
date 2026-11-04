import { cn } from "@/lib/utils";

export function Panel({ className, ...props }: React.ComponentProps<"section">) {
  return (
    <section
      className={cn(
        "min-w-0 rounded-panel border border-border bg-surface shadow-panel",
        className,
      )}
      {...props}
    />
  );
}

export function PanelHeader({
  title,
  subtitle,
  actions,
  className,
}: {
  title: React.ReactNode;
  subtitle?: React.ReactNode;
  actions?: React.ReactNode;
  className?: string;
}) {
  return (
    <header className={cn("flex items-start justify-between gap-3 px-4 pt-4 sm:px-5 sm:pt-5", className)}>
      <div className="min-w-0">
        <h2 className="text-sm font-semibold leading-5 text-fg">{title}</h2>
        {subtitle ? <p className="mt-0.5 text-xs leading-4 text-fg-muted">{subtitle}</p> : null}
      </div>
      {actions ? <div className="flex shrink-0 items-center gap-2">{actions}</div> : null}
    </header>
  );
}

export function PanelBody({ className, ...props }: React.ComponentProps<"div">) {
  return <div className={cn("px-4 pb-4 pt-3 sm:px-5 sm:pb-5", className)} {...props} />;
}
