import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  // Pin tracing root to this app so multi-lockfile workspaces (the EDCOCR
  // monorepo has Python tooling at the root) don't trip the auto-detect.
  outputFileTracingRoot: __dirname,
};

export default nextConfig;
