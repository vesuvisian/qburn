/**
 * Expand file and directory paths into image files or a single video source.
 * Matches Burn CLI rules: explicit files always queued; dirs yield image exts;
 * video containers / `--video` frame dirs are single-video mode.
 */

import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import os from "node:os";

const IMAGE_EXTENSIONS = new Set([
  "jpg",
  "jpeg",
  "png",
  "webp",
  "jpe",
  "jfif",
]);

const VIDEO_EXTENSIONS = new Set(["mp4", "webm", "mov", "mkv", "avi"]);

export const VIDEO_MIN_FRAMES = 4;
/** Keep temporal depth modest — long greedy decode still loops past ~16 frames. */
export const VIDEO_MAX_FRAMES_CLI = 16;
export const VIDEO_DEFAULT_FPS = 2;

export function isImagePath(filePath) {
  const ext = path.extname(filePath).slice(1).toLowerCase();
  return IMAGE_EXTENSIONS.has(ext);
}

export function isVideoPath(filePath) {
  const ext = path.extname(filePath).slice(1).toLowerCase();
  return VIDEO_EXTENSIONS.has(ext);
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
      if (isVideoPath(p)) {
        throw new Error(
          `video file ${p} mixed into image describe — pass it alone`
        );
      }
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

export function collectFrameDir(dir) {
  const out = [];
  collectFromDir(dir, false, out);
  const resolved = [...new Set(out.map((p) => path.resolve(p)))].sort();
  if (resolved.length === 0) {
    throw new Error(
      `no frames found in ${dir} (looking for ${[...IMAGE_EXTENSIONS].join(", ")})`
    );
  }
  return resolved;
}

/**
 * @returns {{ type: "images", paths: string[] } | { type: "video", source: { kind: "file"|"frames", path: string } }}
 */
export function classifyDescribePaths(paths, recursive, forceVideo) {
  if (!paths.length) throw new Error("no paths given");

  if (forceVideo) {
    if (paths.length !== 1) {
      throw new Error("--video expects exactly one path (file or frame directory)");
    }
    const p = paths[0];
    if (!fs.existsSync(p)) throw new Error(`path not found: ${p}`);
    const st = fs.statSync(p);
    if (st.isFile()) {
      if (!isVideoPath(p)) {
        throw new Error(
          `--video file must be a video (${[...VIDEO_EXTENSIONS].join(", ")})`
        );
      }
      return { type: "video", source: { kind: "file", path: path.resolve(p) } };
    }
    if (st.isDirectory()) {
      return { type: "video", source: { kind: "frames", path: path.resolve(p) } };
    }
    throw new Error(`not a file or directory: ${p}`);
  }

  const videoFiles = paths.filter((p) => fs.existsSync(p) && fs.statSync(p).isFile() && isVideoPath(p));
  if (videoFiles.length) {
    if (videoFiles.length !== 1 || paths.length !== 1) {
      throw new Error(
        `single-video describe takes exactly one video file (found ${videoFiles.length}); do not mix with images`
      );
    }
    return {
      type: "video",
      source: { kind: "file", path: path.resolve(videoFiles[0]) },
    };
  }

  return { type: "images", paths: collectImages(paths, recursive) };
}

/** Uniform sample indices (HF Qwen3-VL style). */
export function sampleFrameIndices(
  totalFrames,
  srcFps,
  { fps = null, numFrames = null, minFrames = VIDEO_MIN_FRAMES, maxFrames = VIDEO_MAX_FRAMES_CLI } = {}
) {
  if (!totalFrames) throw new Error("video has zero frames");
  if (fps != null && numFrames != null) {
    throw new Error("`fps` and `numFrames` are mutually exclusive");
  }
  let n;
  if (numFrames != null) {
    n = Math.max(1, numFrames | 0);
  } else {
    const useFps = fps == null ? VIDEO_DEFAULT_FPS : fps;
    const src = srcFps > 0 ? srcFps : 24;
    n = Math.max(1, Math.round((totalFrames / src) * useFps));
  }
  n = Math.max(n, Math.min(minFrames, totalFrames));
  n = Math.min(n, maxFrames, totalFrames);
  n = Math.max(n, 1);
  if (n === 1) return [0];
  const indices = [];
  for (let i = 0; i < n; i++) {
    indices.push(
      Math.min(totalFrames - 1, Math.round((i * (totalFrames - 1)) / (n - 1)))
    );
  }
  return indices;
}

function whichFfmpeg() {
  const r = spawnSync("ffmpeg", ["-version"], { encoding: "utf8" });
  if (r.status !== 0) {
    throw new Error("ffmpeg not found on PATH (required for video container files)");
  }
}

/**
 * Probe container duration + fps. Prefer format duration over packet counts —
 * `nb_read_packets` can overshoot the real timeline and send seeks past EOF.
 * @returns {{ duration: number, srcFps: number, totalFrames: number }}
 */
function probeVideoMeta(videoPath) {
  const probe = spawnSync(
    "ffprobe",
    [
      "-v",
      "error",
      "-select_streams",
      "v:0",
      "-show_entries",
      "stream=r_frame_rate,avg_frame_rate,nb_frames:format=duration",
      "-of",
      "json",
    ],
    { encoding: "utf8" }
  );
  if (probe.status === 0 && probe.stdout) {
    try {
      const v = JSON.parse(probe.stdout);
      const stream = v.streams?.[0] || {};
      const fps =
        parseRate(stream.avg_frame_rate) ||
        parseRate(stream.r_frame_rate) ||
        24;
      const duration = Number(v.format?.duration);
      if (Number.isFinite(duration) && duration > 0) {
        const fromDur = Math.max(1, Math.round(duration * fps));
        const nb = Number(stream.nb_frames || 0);
        // Prefer duration-derived count; only trust nb_frames if it fits timeline.
        const totalFrames =
          nb > 0 && nb <= fromDur * 1.05 ? nb : fromDur;
        return { duration, srcFps: fps, totalFrames };
      }
    } catch {
      /* fall through */
    }
  }
  // Fallback: ffmpeg -i stderr
  const ff = spawnSync("ffmpeg", ["-hide_banner", "-i", videoPath], {
    encoding: "utf8",
  });
  const stderr = ff.stderr || "";
  let fps = 24;
  let duration = null;
  for (const line of stderr.split("\n")) {
    if (line.includes("Duration:")) {
      const m = line.match(/Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)/);
      if (m) {
        duration = Number(m[1]) * 3600 + Number(m[2]) * 60 + Number(m[3]);
      }
    }
    if (line.includes("Video:")) {
      const m = line.match(/([\d.]+)\s*fps/);
      if (m) fps = Number(m[1]);
    }
  }
  if (duration == null || !(duration > 0)) {
    throw new Error("could not probe video duration (install ffprobe)");
  }
  return {
    duration,
    srcFps: fps,
    totalFrames: Math.max(1, Math.round(duration * fps)),
  };
}

/** Uniform sample timestamps in [0, duration), clamped off the EOF edge. */
export function sampleTimestamps(
  duration,
  { fps = null, numFrames = null, minFrames = VIDEO_MIN_FRAMES, maxFrames = VIDEO_MAX_FRAMES_CLI } = {}
) {
  if (!(duration > 0)) throw new Error("video has invalid duration");
  if (fps != null && numFrames != null) {
    throw new Error("`fps` and `numFrames` are mutually exclusive");
  }
  let n;
  if (numFrames != null) {
    n = Math.max(1, numFrames | 0);
  } else {
    const useFps = fps == null ? VIDEO_DEFAULT_FPS : fps;
    n = Math.max(1, Math.round(duration * useFps));
  }
  n = Math.max(n, minFrames);
  n = Math.min(n, maxFrames);
  n = Math.max(n, 1);

  // Stay clear of EOF — late `-ss` often yields "No filtered frames" with exit 0.
  const end = Math.max(0, duration - 0.15);
  if (n === 1) return [Math.min(end / 2, end)];
  const times = [];
  for (let i = 0; i < n; i++) {
    times.push(Math.min(end, (i * end) / (n - 1)));
  }
  return times;
}

/** Extract one PNG at timestamp `t`. Fast pre-input seek, then accurate fallback. */
function extractFrameAt(videoPath, t, dest) {
  const attempt = (ssBeforeInput) => {
    const args = ["-hide_banner", "-loglevel", "error", "-y"];
    if (ssBeforeInput) {
      args.push("-ss", t.toFixed(6), "-i", videoPath);
    } else {
      // Accurate (slower): decode then seek.
      args.push("-i", videoPath, "-ss", t.toFixed(6));
    }
    args.push("-frames:v", "1", "-update", "1", dest);
    return spawnSync("ffmpeg", args, { encoding: "utf8" });
  };

  let r = attempt(true);
  if (r.status === 0 && fs.existsSync(dest) && fs.statSync(dest).size > 0) {
    return;
  }
  try {
    fs.unlinkSync(dest);
  } catch {
    /* ignore */
  }
  r = attempt(false);
  if (r.status !== 0 || !fs.existsSync(dest) || fs.statSync(dest).size === 0) {
    throw new Error(
      `ffmpeg wrote no pixels for frame at t=${t.toFixed(3)}s${
        r.stderr ? `: ${String(r.stderr).trim()}` : ""
      }`
    );
  }
}

function parseRate(s) {
  if (!s || s === "0/0") return null;
  if (s.includes("/")) {
    const [a, b] = s.split("/").map(Number);
    if (!b) return null;
    return a / b;
  }
  const n = Number(s);
  return Number.isFinite(n) ? n : null;
}

/**
 * @returns {{ paths: string[], timestamps: number[] }} absolute PNG paths + seek times
 */
export function loadVideoFramePaths(source, { fps = null, numFrames = null } = {}) {
  if (source.kind === "frames") {
    const srcFps = 24;
    const all = collectFrameDir(source.path);
    const indices = sampleFrameIndices(all.length, srcFps, {
      fps: fps ?? (numFrames == null ? VIDEO_DEFAULT_FPS : null),
      numFrames,
    });
    return {
      paths: indices.map((i) => all[i]),
      timestamps: indices.map((i) => i / srcFps),
    };
  }

  whichFfmpeg();
  const { duration } = probeVideoMeta(source.path);
  const times = sampleTimestamps(duration, {
    fps: fps ?? (numFrames == null ? VIDEO_DEFAULT_FPS : null),
    numFrames,
  });

  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "qburn-tjs-video-"));
  const out = [];
  for (let i = 0; i < times.length; i++) {
    const dest = path.join(tmp, `frame-${String(i).padStart(5, "0")}.png`);
    extractFrameAt(source.path, times[i], dest);
    out.push(dest);
  }
  return { paths: out, timestamps: times };
}
