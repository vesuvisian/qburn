/**
 * Durable ONNX weight cache for qburn-tjs.
 *
 * transformers.js defaults to `<package>/.cache` inside node_modules, which
 * disappears on reinstall. We prefer a repo-local dir under weights/ (already
 * gitignored), with an env override.
 */

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/** Repo root: scripts/tjs-cli/lib → ../../.. */
export const REPO_ROOT = path.resolve(__dirname, "../../..");

/** Default cache: weights/tjs-cache (alongside burnpacks / HF checkout). */
export const DEFAULT_CACHE_DIR = path.join(REPO_ROOT, "weights", "tjs-cache");

/**
 * Resolve the filesystem cache directory.
 *
 * Precedence:
 *   1. QBURN_TJS_CACHE
 *   2. TRANSFORMERS_CACHE (HF / transformers.js convention)
 *   3. weights/tjs-cache under the repo
 */
export function resolveCacheDir({
  env = process.env,
  defaultDir = DEFAULT_CACHE_DIR,
} = {}) {
  const fromEnv =
    env.QBURN_TJS_CACHE?.trim() || env.TRANSFORMERS_CACHE?.trim() || "";
  if (fromEnv) {
    return path.resolve(expandHome(fromEnv));
  }
  return path.resolve(defaultDir);
}

/** Legacy location used by transformers.js before we overrode cacheDir. */
export const LEGACY_CACHE_DIR = path.join(
  REPO_ROOT,
  "scripts/tjs-cli/node_modules/@huggingface/transformers/.cache"
);

/**
 * Move a previous node_modules/.cache tree into the durable cache dir when
 * the destination is empty. No-op if either side is missing / already populated.
 * @returns {boolean} true if a migration ran
 */
export function migrateLegacyCache(cacheDir, legacyDir = LEGACY_CACHE_DIR) {
  if (!fs.existsSync(legacyDir)) return false;
  fs.mkdirSync(cacheDir, { recursive: true });
  const destEntries = fs.readdirSync(cacheDir);
  if (destEntries.length > 0) return false;
  const srcEntries = fs.readdirSync(legacyDir);
  if (srcEntries.length === 0) return false;

  for (const name of srcEntries) {
    fs.renameSync(path.join(legacyDir, name), path.join(cacheDir, name));
  }
  return true;
}

/**
 * Ensure cache dir exists and point transformers.js `env` at it.
 * @param {import("@huggingface/transformers").env} transformersEnv
 * @returns {{ cacheDir: string, migrated: boolean }}
 */
export function configureWeightCache(transformersEnv, options = {}) {
  const cacheDir = resolveCacheDir(options);
  fs.mkdirSync(cacheDir, { recursive: true });
  const migrated =
    cacheDir === path.resolve(DEFAULT_CACHE_DIR)
      ? migrateLegacyCache(cacheDir)
      : false;
  transformersEnv.useFSCache = true;
  transformersEnv.cacheDir = cacheDir;
  return { cacheDir, migrated };
}

function expandHome(p) {
  if (p === "~") return os.homedir();
  if (p.startsWith("~/")) return path.join(os.homedir(), p.slice(2));
  return p;
}
