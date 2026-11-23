/**
 * Shared browser video frame sampler for Burn + transformers.js demos.
 *
 * Seeks a `<video>` element and draws frames to a canvas, exporting JPEG blobs.
 * Matches HF-style uniform sampling with fps / num_frames (mutually exclusive).
 *
 * Seeks must wait for a decoded frame (`requestVideoFrameCallback` when
 * available). Listening for `seeked` alone often captures the previous frame,
 * which makes every sample identical and the VL model invent abstract
 * "digital image" captions.
 */

export const VIDEO_MIN_FRAMES = 4;
export const VIDEO_MAX_FRAMES_WEB = 8;
export const VIDEO_DEFAULT_FPS = 2;

const VIDEO_EXT = /\.(mp4|webm|mov|mkv|avi|m4v)$/i;
const IMAGE_EXT = /\.(png|jpe?g|webp|gif|bmp)$/i;

/**
 * True when the File looks like a video (MIME and/or extension).
 * Some browsers leave `file.type` empty for local `.mp4` drops.
 * @param {File|Blob&{name?: string}} file
 */
export function isVideoFile(file) {
  if (file?.type?.startsWith("video/")) return true;
  const name = typeof file?.name === "string" ? file.name : "";
  return VIDEO_EXT.test(name);
}

/**
 * True when the File looks like an image (MIME and/or extension).
 * @param {File|Blob&{name?: string}} file
 */
export function isImageFile(file) {
  if (file?.type?.startsWith("image/")) return true;
  const name = typeof file?.name === "string" ? file.name : "";
  return IMAGE_EXT.test(name);
}

/**
 * @param {number} totalFrames
 * @param {number} srcFps
 * @param {{ fps?: number|null, numFrames?: number|null, minFrames?: number, maxFrames?: number }} [opts]
 * @returns {number[]}
 */
export function sampleFrameIndices(
  totalFrames,
  srcFps,
  {
    fps = null,
    numFrames = null,
    minFrames = VIDEO_MIN_FRAMES,
    maxFrames = VIDEO_MAX_FRAMES_WEB,
  } = {}
) {
  if (!totalFrames || totalFrames < 1) {
    throw new Error("video has zero frames");
  }
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

/**
 * Wait until `video` has buffered through at least `t` seconds (best-effort).
 * @param {HTMLVideoElement} video
 * @param {number} t
 * @param {number} [timeoutMs]
 */
async function waitForBuffered(video, t, timeoutMs = 4000) {
  const covered = () => {
    const b = video.buffered;
    for (let i = 0; i < b.length; i++) {
      if (b.start(i) <= t + 1e-3 && b.end(i) >= Math.min(t + 0.05, video.duration)) {
        return true;
      }
    }
    // Blob / fully-local sources often report a single range [0, duration].
    return b.length > 0 && b.start(0) <= 1e-3 && b.end(b.length - 1) >= t;
  };
  if (covered()) return;
  await new Promise((resolve) => {
    const done = () => {
      video.removeEventListener("progress", onProgress);
      clearTimeout(timer);
      resolve();
    };
    const onProgress = () => {
      if (covered()) done();
    };
    const timer = setTimeout(done, timeoutMs);
    video.addEventListener("progress", onProgress);
    onProgress();
  });
}

/**
 * Seek then wait until a decoded frame for that time is presentable.
 * @param {HTMLVideoElement} video
 * @param {number} t
 */
async function seekAndPresent(video, t) {
  await waitForBuffered(video, t);
  await new Promise((resolve, reject) => {
    const onSeeked = () => {
      cleanup();
      resolve();
    };
    const onError = () => {
      cleanup();
      reject(new Error("video seek failed"));
    };
    const cleanup = () => {
      video.removeEventListener("seeked", onSeeked);
      video.removeEventListener("error", onError);
    };
    // Listener before currentTime — otherwise a sync/microtask seeked is missed
    // and the promise hangs (or we draw a stale frame after a later event).
    video.addEventListener("seeked", onSeeked);
    video.addEventListener("error", onError);
    try {
      video.currentTime = t;
    } catch (err) {
      cleanup();
      reject(err);
    }
  });

  // `seeked` can fire before the new frame is painted into the element.
  if (typeof video.requestVideoFrameCallback === "function") {
    await Promise.race([
      new Promise((resolve) => {
        const id = video.requestVideoFrameCallback(() => resolve());
        // Keep a handle so engines that require cancel stay happy if we time out.
        setTimeout(() => {
          if (typeof video.cancelVideoFrameCallback === "function") {
            try {
              video.cancelVideoFrameCallback(id);
            } catch {
              /* ignore */
            }
          }
        }, 250);
      }),
      new Promise((resolve) => setTimeout(resolve, 200)),
    ]);
  } else {
    await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
    await new Promise((r) => setTimeout(r, 32));
  }
}

/**
 * Sample frames from a File/Blob/URL into JPEG ArrayBuffers.
 *
 * @param {File|Blob|string} src
 * @param {{ fps?: number|null, numFrames?: number|null, quality?: number }} [opts]
 * @returns {Promise<{ frameBytes: ArrayBuffer[], timestamps: number[], previewUrl: string, duration: number, revokePreview: () => void }>}
 */
export async function sampleVideoFrames(src, opts = {}) {
  const quality = opts.quality ?? 0.92;
  // Progressive HTTP ranges often only buffer the start; seeking then redraws
  // the same keyframe. Prefer a fully-local blob URL for string sources.
  let objectUrl;
  let revokeObject = false;
  if (typeof src === "string") {
    if (src.startsWith("blob:")) {
      objectUrl = src;
    } else {
      const resp = await fetch(src);
      if (!resp.ok) {
        throw new Error(`failed to fetch video (${resp.status})`);
      }
      const blob = await resp.blob();
      objectUrl = URL.createObjectURL(blob);
      revokeObject = true;
    }
  } else {
    objectUrl = URL.createObjectURL(src);
    revokeObject = true;
  }

  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.preload = "auto";
  video.src = objectUrl;

  await new Promise((resolve, reject) => {
    video.onloadeddata = () => resolve();
    video.onerror = () => reject(new Error("failed to load video"));
  });

  // Prime the decoder; some engines won't present seek targets otherwise.
  try {
    await video.play();
  } catch {
    /* autoplay / play() optional */
  }
  video.pause();

  const duration = video.duration;
  if (!Number.isFinite(duration) || duration <= 0) {
    if (revokeObject) URL.revokeObjectURL(objectUrl);
    throw new Error("video has invalid duration");
  }

  // Estimate total frames from a nominal 30 fps timeline when metadata lacks it.
  const srcFps = 30;
  const totalFrames = Math.max(1, Math.round(duration * srcFps));
  const indices = sampleFrameIndices(totalFrames, srcFps, {
    fps: opts.fps ?? null,
    numFrames: opts.numFrames ?? null,
    minFrames: VIDEO_MIN_FRAMES,
    maxFrames: VIDEO_MAX_FRAMES_WEB,
  });

  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 360;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) {
    if (revokeObject) URL.revokeObjectURL(objectUrl);
    throw new Error("canvas 2d unavailable");
  }

  const frameBytes = [];
  const timestamps = [];
  const end = Math.max(0, duration - 0.15);
  for (const frameIdx of indices) {
    const t = Math.min(end, Math.max(0, frameIdx / srcFps));
    await seekAndPresent(video, t);
    timestamps.push(video.currentTime);
    // createImageBitmap tends to grab the presented frame more reliably than
    // drawImage(video) alone on some engines.
    let bitmap = null;
    try {
      bitmap = await createImageBitmap(video);
      ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    } finally {
      bitmap?.close?.();
    }
    const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    let sum = 0;
    let sumSq = 0;
    const nPix = pixels.length / 4;
    for (let p = 0; p < pixels.length; p += 4) {
      const y = (pixels[p] + pixels[p + 1] + pixels[p + 2]) / 3;
      sum += y;
      sumSq += y * y;
    }
    const mean = sum / nPix;
    const variance = sumSq / nPix - mean * mean;
    if (mean < 2 && variance < 1) {
      video.removeAttribute("src");
      video.load();
      if (revokeObject) URL.revokeObjectURL(objectUrl);
      throw new Error(
        "sampled a blank video frame — the browser could not decode this seek"
      );
    }
    const blob = await new Promise((resolve, reject) => {
      canvas.toBlob(
        (b) => (b ? resolve(b) : reject(new Error("toBlob failed"))),
        "image/jpeg",
        quality
      );
    });
    frameBytes.push(await blob.arrayBuffer());
  }

  // Reject obviously-broken samples (every JPEG identical → stale draws).
  if (frameBytes.length >= 2) {
    const sizes = frameBytes.map((b) => b.byteLength);
    const allSameSize = sizes.every((s) => s === sizes[0]);
    if (allSameSize) {
      const first = new Uint8Array(frameBytes[0]);
      const identical = frameBytes.every((buf) => {
        const u8 = new Uint8Array(buf);
        if (u8.length !== first.length) return false;
        for (let i = 0; i < u8.length; i += Math.max(1, (u8.length / 64) | 0)) {
          if (u8[i] !== first[i]) return false;
        }
        return true;
      });
      if (identical) {
        video.removeAttribute("src");
        video.load();
        if (revokeObject) URL.revokeObjectURL(objectUrl);
        throw new Error(
          "video frame sampling produced identical frames — try another browser or re-encode the clip"
        );
      }
    }
  }

  video.removeAttribute("src");
  video.load();
  // Keep object URL for preview (caller revokes via revokePreview).
  const previewUrl = objectUrl;

  return {
    frameBytes,
    timestamps,
    previewUrl,
    duration,
    revokePreview: () => {
      if (revokeObject) URL.revokeObjectURL(objectUrl);
    },
  };
}

/**
 * Average frame timestamps within each temporal patch (HF Qwen3-VL).
 * @param {number[]} frameTimes
 * @param {number} [temporalPatchSize=2]
 * @returns {number[]}
 */
export function temporalGroupTimestamps(frameTimes, temporalPatchSize = 2) {
  const tps = Math.max(1, temporalPatchSize | 0);
  if (!frameTimes.length) return [];
  const times = frameTimes.slice();
  while (times.length % tps !== 0) {
    times.push(times[times.length - 1]);
  }
  const out = [];
  for (let i = 0; i < times.length; i += tps) {
    out.push((times[i] + times[i + tps - 1]) / 2);
  }
  return out;
}
