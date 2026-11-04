/** Sidebar preference, shared by the server layout (reads the cookie) and the client shell (writes it). */
export type NavPref = "expanded" | "collapsed" | null;
export const NAV_COOKIE = "uiq_nav";
