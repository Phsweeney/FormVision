import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Emits `.next/standalone` — a self-contained server plus only the
  // node_modules actually imported. This is what the Dockerfile copies, and it
  // is the difference between a lean image and one carrying the whole
  // dependency tree.
  //
  // The standalone server does not include `public` or `.next/static`; the
  // Dockerfile copies those in explicitly, as the Next.js docs describe.
  output: "standalone",
};

export default nextConfig;
