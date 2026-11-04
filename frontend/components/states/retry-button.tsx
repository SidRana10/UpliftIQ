"use client";

import { RotateCw } from "lucide-react";
import { useRouter } from "next/navigation";
import { useTransition } from "react";
import { Button } from "@/components/ui/button";

export function RetryButton() {
  const router = useRouter();
  const [pending, start] = useTransition();
  return (
    <Button variant="secondary" size="sm" disabled={pending} onClick={() => start(() => router.refresh())}>
      <RotateCw className={pending ? "animate-spin motion-reduce:animate-none" : ""} />
      {pending ? "Retrying" : "Retry"}
    </Button>
  );
}
