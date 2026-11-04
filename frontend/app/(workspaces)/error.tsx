"use client";

import { SectionError } from "@/components/states/section-error";

export default function WorkspaceError({ error }: { error: Error & { digest?: string } }) {
  return (
    <SectionError
      what="This page"
      error={{ kind: "server", status: 500, message: error.message || "Unexpected error while rendering." }}
    />
  );
}
