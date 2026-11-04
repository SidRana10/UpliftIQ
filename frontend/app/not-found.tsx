import Link from "next/link";
import { Button } from "@/components/ui/button";

export default function NotFound() {
  return (
    <main className="flex min-h-dvh flex-col items-center justify-center gap-4 px-6 text-center">
      <p className="eyebrow">404</p>
      <h1 className="text-2xl font-semibold text-fg">That page doesn&apos;t exist</h1>
      <Button asChild variant="primary">
        <Link href="/overview">Go to the Overview</Link>
      </Button>
    </main>
  );
}
