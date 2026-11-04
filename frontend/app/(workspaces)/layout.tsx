import { cookies } from "next/headers";
import { AppShell } from "@/components/shell/app-shell";
import { NAV_COOKIE, type NavPref } from "@/lib/nav";

export default async function WorkspacesLayout({ children }: { children: React.ReactNode }) {
  const stored = (await cookies()).get(NAV_COOKIE)?.value;
  const initialNav: NavPref = stored === "collapsed" || stored === "expanded" ? stored : null;
  return <AppShell initialNav={initialNav}>{children}</AppShell>;
}
