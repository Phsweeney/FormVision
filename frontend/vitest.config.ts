import { fileURLToPath } from "node:url";

import { defineConfig } from "vitest/config";

/**
 * Vitest configuration.
 *
 * The live webcam mode runs the analysis engine (geometry, smoothing, rep
 * detection, metrics, coaching) in the browser as pure TypeScript. Those
 * modules are framework-free by design, exactly like their Python
 * counterparts, so they are tested in a plain Node environment with no jsdom
 * and no React. This mirrors the backend, whose analysis suite runs without
 * MediaPipe or a real video.
 */
export default defineConfig({
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
  resolve: {
    // Match the `@/*` path alias from tsconfig so tests import the same way
    // the app does.
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
});
