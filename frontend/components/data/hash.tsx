"use client";

import { Check, Copy } from "lucide-react";
import { useState } from "react";
import { shortHash } from "@/lib/format";

/** Truncated checksum in mono with a copy button; the full value is the tooltip/title. */
export function Hash({ value, label }: { value: string; label?: string }) {
  const [copied, setCopied] = useState(false);
  async function copy() {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    } catch {
      // clipboard unavailable (insecure context): leave the value selectable
    }
  }
  return (
    <span className="inline-flex items-center gap-1.5">
      <code title={value} className="font-mono text-xs text-fg-secondary">
        {shortHash(value)}
      </code>
      <button
        type="button"
        onClick={copy}
        aria-label={`Copy ${label ?? "checksum"}`}
        className="inline-flex size-5 items-center justify-center rounded text-fg-muted transition-colors hover:bg-subtle hover:text-fg"
      >
        {copied ? <Check className="size-3.5 text-teal" /> : <Copy className="size-3.5" />}
      </button>
    </span>
  );
}
