import type { NextConfig } from "next";

const api = process.env.UPLIFTIQ_API_URL ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  devIndicators: false,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
  // Same-origin proxy: the browser only ever talks to this app; no CORS dependency.
  async rewrites() {
    return [{ source: "/api/v1/:path*", destination: `${api}/api/v1/:path*` }];
  },
};

export default nextConfig;
