/**
 * Expand file and directory paths into a sorted, deduped list of image files.
 * Matches Burn CLI rules: explicit files always queued; dirs yield image exts.
 */

import fs from "node:fs";
import path from "node:path";

const IMAGE_EXTENSIONS = new Set([
  "jpg",
  "jpeg",
  "png",
  "webp",
  "jpe",
  "jfif",
]);

export function isImagePath(filePath) {
  const ext = path.extname(filePath).slice(1).toLowerCase();
  return IMAGE_EXTENSIONS.has(ext);
}

function collectFromDir(dir, recursive, out) {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  entries.sort((a, b) => a.name.localeCompare(b.name));
  for (const entry of entries) {
    const full = path.join(dir, entry.name);
    if (entry.isFile()) {
      if (isImagePath(full)) out.push(full);
    } else if (recursive && entry.isDirectory()) {
      collectFromDir(full, true, out);
    }
  }
}

/**
 * @param {string[]} paths
 * @param {boolean} recursive
 * @returns {string[]}
 */
export function collectImages(paths, recursive) {
  const out = [];
  for (const p of paths) {
    if (!fs.existsSync(p)) {
      throw new Error(`path not found: ${p}`);
    }
    const st = fs.statSync(p);
    if (st.isFile()) {
      out.push(path.resolve(p));
    } else if (st.isDirectory()) {
      collectFromDir(p, recursive, out);
    } else {
      throw new Error(`not a file or directory: ${p}`);
    }
  }
  const resolved = [...new Set(out.map((p) => path.resolve(p)))].sort();
  if (resolved.length === 0) {
    throw new Error(
      `no images found (looking for ${[...IMAGE_EXTENSIONS].join(", ")}${
        recursive ? "; recursive" : ""
      })`
    );
  }
  return resolved;
}
