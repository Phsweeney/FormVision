/**
 * Populate the MediaPipe runtime assets the live webcam mode needs.
 *
 * Two things have to sit under `public/` so the browser can fetch them from our
 * own origin rather than a third-party CDN:
 *
 *   public/mediapipe/wasm/   the tasks-vision WASM fileset (shipped in the npm
 *                            package; copied out of node_modules)
 *   public/models/           the pose landmarker bundle (reused from the
 *                            backend's copy, or downloaded once)
 *
 * Both are large binaries, so they are gitignored, exactly like the backend's
 * `models/` directory. This script recreates them from sources that are already
 * on the machine (or one download), and is idempotent: assets already present
 * are left alone.
 *
 * It is deliberately NON-FATAL. If it cannot fetch the model (offline, no
 * backend copy), it warns and exits 0 so the upload mode and the production
 * build are never blocked by a live-mode asset.
 */

import { existsSync } from "node:fs";
import { copyFile, mkdir, readdir, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const frontendRoot = join(here, "..");
const repoRoot = join(frontendRoot, "..");

const MODEL_FILE = "pose_landmarker_lite.task";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/" +
  "pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

async function copyWasm() {
  const src = join(
    frontendRoot,
    "node_modules",
    "@mediapipe",
    "tasks-vision",
    "wasm",
  );
  const dest = join(frontendRoot, "public", "mediapipe", "wasm");
  if (!existsSync(src)) {
    console.warn(
      "[setup-live] @mediapipe/tasks-vision not installed; run npm install first.",
    );
    return;
  }
  await mkdir(dest, { recursive: true });
  for (const name of await readdir(src)) {
    await copyFile(join(src, name), join(dest, name));
  }
  console.log("[setup-live] WASM fileset ready at public/mediapipe/wasm");
}

async function ensureModel() {
  const dest = join(frontendRoot, "public", "models", MODEL_FILE);
  if (existsSync(dest)) {
    console.log("[setup-live] model already present; skipping.");
    return;
  }
  await mkdir(dirname(dest), { recursive: true });

  // Prefer the copy the backend already downloaded: no network needed.
  const backendCopy = join(repoRoot, "backend", "models", MODEL_FILE);
  if (existsSync(backendCopy)) {
    await copyFile(backendCopy, dest);
    console.log("[setup-live] model copied from backend/models.");
    return;
  }

  console.log("[setup-live] downloading pose landmarker model...");
  const response = await fetch(MODEL_URL);
  if (!response.ok) {
    throw new Error(`model download failed: HTTP ${response.status}`);
  }
  await writeFile(dest, Buffer.from(await response.arrayBuffer()));
  console.log("[setup-live] model downloaded to public/models.");
}

try {
  await copyWasm();
  await ensureModel();
} catch (error) {
  console.warn(
    `[setup-live] could not prepare live assets (${error.message}). ` +
      "Upload mode and the build are unaffected; live mode needs these assets.",
  );
}
