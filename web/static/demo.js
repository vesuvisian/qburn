/**
 * qburn browser demo — WebGPU gate, weight load via worker, streaming caption.
 *
 * Manifest resolution (first match wins):
 *   1. `?manifest=` query param
 *   2. `window.QBURN_MANIFEST` (set before this script)
 *   3. `./weights/manifest.json` when served locally via `preview.py --weights`
 *   4. Hugging Face Hub default
 *
 * Media: up to 4 images **or** one video (exclusive). Video frames are sampled
 * in the main thread (`video-sample.js`) then sent to the worker.
 */

import {
  isImageFile,
  isVideoFile,
  sampleVideoFrames,
  VIDEO_MIN_FRAMES,
} from "./video-sample.js";

const HUB_MANIFEST =
  "https://huggingface.co/vesuvisian/qburn-weights/resolve/main/manifest.json";

const MAX_IMAGES = 4;

const gate = document.getElementById("gate");
const statusEl = document.getElementById("status");
const progress = document.getElementById("progress");
const barFill = document.getElementById("barFill");
const progressLabel = document.getElementById("progressLabel");
const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const thumbs = document.getElementById("thumbs");
const promptEl = document.getElementById("prompt");
const maxTokensEl = document.getElementById("maxTokens");
const temperatureEl = document.getElementById("temperature");
const describeBtn = document.getElementById("describeBtn");
const caption = document.getElementById("caption");

let worker = null;
/** @type {{ kind: "image", bytes: Uint8Array, url: string }[]} */
let selectedImages = [];
/**
 * @type {null | {
 *   kind: "video",
 *   frameBytes: Uint8Array[],
 *   frameUrls: string[],
 *   timestamps: number[],
 *   url: string,
 *   revokePreview: () => void,
 * }}
 */
let selectedVideo = null;
let ready = false;
let busy = false;
let manifestLabel = "hub";
let generatePulse = null;
/** Prefill top-k dump (extra GPU readback). Enable with `?probe=1`. */
const probeEnabled = (() => {
  const v = new URLSearchParams(location.search).get("probe");
  return v === "1" || v === "true";
})();

function setStatus(msg) {
  statusEl.textContent = msg;
}

function setProgress(frac, label) {
  progress.classList.remove("hidden");
  const pct = Math.max(0, Math.min(100, Math.round(frac * 100)));
  barFill.style.width = `${pct}%`;
  progressLabel.textContent = label || `${pct}%`;
}

function hideProgress() {
  progress.classList.add("hidden");
}

function requireWebGpu() {
  if (!navigator.gpu) {
    gate.textContent =
      "WebGPU is required. Use a recent Chrome/Edge (or Safari with WebGPU enabled).";
    gate.classList.remove("hidden");
    setStatus("WebGPU unavailable");
    return false;
  }
  return true;
}

async function localManifestAvailable() {
  const url = new URL("./weights/manifest.json", location.href).href;
  try {
    const res = await fetch(url, { method: "HEAD", cache: "no-store" });
    if (res.ok) return url;
    const get = await fetch(url, { method: "GET", cache: "no-store" });
    if (get.ok) return url;
  } catch {
    /* not served */
  }
  return null;
}

async function resolveManifestUrl() {
  const params = new URLSearchParams(location.search);
  const fromQuery = params.get("manifest");
  if (fromQuery) {
    manifestLabel = "query";
    return fromQuery;
  }
  if (typeof window.QBURN_MANIFEST === "string" && window.QBURN_MANIFEST) {
    manifestLabel = "config";
    return window.QBURN_MANIFEST;
  }
  const local = await localManifestAvailable();
  if (local) {
    manifestLabel = "local";
    return local;
  }
  manifestLabel = "hub";
  return HUB_MANIFEST;
}

function hasMedia() {
  return selectedImages.length > 0 || selectedVideo != null;
}

function syncDescribeEnabled() {
  describeBtn.disabled = !ready || !hasMedia() || busy;
}

function clearImages() {
  for (const item of selectedImages) {
    if (item.url) URL.revokeObjectURL(item.url);
  }
  selectedImages = [];
}

function clearVideo() {
  if (selectedVideo) {
    for (const url of selectedVideo.frameUrls || []) {
      URL.revokeObjectURL(url);
    }
    selectedVideo.revokePreview?.();
    selectedVideo = null;
  }
}

function renderThumbs() {
  thumbs.replaceChildren();
  if (selectedVideo) {
    const removeVideo = () => {
      clearVideo();
      renderThumbs();
      caption.textContent = "";
    };
    // Show the frames the model will actually see (not just the <video>).
    selectedVideo.frameUrls.forEach((url, index) => {
      const wrap = document.createElement("div");
      wrap.className = "thumb";
      const img = document.createElement("img");
      img.src = url;
      img.alt = `Video frame ${index + 1}`;
      if (index === 0) {
        const remove = document.createElement("button");
        remove.type = "button";
        remove.setAttribute("aria-label", "Remove video");
        remove.textContent = "×";
        remove.addEventListener("click", (e) => {
          e.preventDefault();
          e.stopPropagation();
          removeVideo();
        });
        wrap.append(img, remove);
      } else {
        wrap.append(img);
      }
      thumbs.append(wrap);
    });
  } else {
    selectedImages.forEach((item, index) => {
      const wrap = document.createElement("div");
      wrap.className = "thumb";
      const img = document.createElement("img");
      img.src = item.url;
      img.alt = `Image ${index + 1}`;
      const remove = document.createElement("button");
      remove.type = "button";
      remove.setAttribute("aria-label", `Remove image ${index + 1}`);
      remove.textContent = "×";
      remove.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        const [removed] = selectedImages.splice(index, 1);
        if (removed?.url) URL.revokeObjectURL(removed.url);
        renderThumbs();
        caption.textContent = "";
      });
      wrap.append(img, remove);
      thumbs.append(wrap);
    });
  }
  syncDescribeEnabled();
}

async function ingestFiles(fileList) {
  const files = [...fileList];
  const videos = files.filter((f) => isVideoFile(f));
  const images = files.filter((f) => isImageFile(f) && !isVideoFile(f));

  if (videos.length && images.length) {
    setStatus("Choose either one video or up to 4 images — not both");
    return;
  }

  if (videos.length) {
    if (videos.length > 1) {
      setStatus("Only one video at a time");
      return;
    }
    clearImages();
    clearVideo();
    setStatus("Sampling video frames…");
    try {
      // Fixed frame count (not fps) keeps spatial resolution under the HF
      // video pixel budget — 8 frames at 2fps crushes detail and the model
      // falls back to vague "abstract digital image" captions.
      const sampled = await sampleVideoFrames(videos[0], {
        numFrames: VIDEO_MIN_FRAMES,
      });
      if (sampled.frameBytes.length < VIDEO_MIN_FRAMES) {
        sampled.revokePreview?.();
        setStatus(`Need at least ${VIDEO_MIN_FRAMES} frames from the video`);
        return;
      }
      const frameBytes = sampled.frameBytes.map((ab) => new Uint8Array(ab));
      selectedVideo = {
        kind: "video",
        frameBytes,
        frameUrls: frameBytes.map((bytes) =>
          URL.createObjectURL(new Blob([bytes], { type: "image/jpeg" }))
        ),
        timestamps: sampled.timestamps.slice(),
        url: sampled.previewUrl,
        revokePreview: sampled.revokePreview,
      };
      setStatus(
        `Video ready (${selectedVideo.frameBytes.length} frames sampled)`
      );
    } catch (err) {
      setStatus(err?.message || "video sample failed");
      return;
    }
    renderThumbs();
    caption.textContent = "";
    fileInput.value = "";
    return;
  }

  if (!images.length) return;
  if (selectedVideo) {
    clearVideo();
  }
  const room = MAX_IMAGES - selectedImages.length;
  if (room <= 0) {
    setStatus(`Already at ${MAX_IMAGES} images (remove one to add more)`);
    return;
  }
  const take = images.slice(0, room);
  for (const file of take) {
    const bytes = new Uint8Array(await file.arrayBuffer());
    selectedImages.push({ kind: "image", bytes, url: URL.createObjectURL(file) });
  }
  if (images.length > take.length) {
    setStatus(`Added ${take.length}; max is ${MAX_IMAGES} images`);
  }
  renderThumbs();
  caption.textContent = "";
  fileInput.value = "";
}

function bindUi() {
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag"));
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag");
    if (e.dataTransfer?.files?.length) ingestFiles(e.dataTransfer.files);
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files?.length) ingestFiles(fileInput.files);
  });
  describeBtn.addEventListener("click", () => {
    if (!ready || !hasMedia() || busy) return;
    runDescribe();
  });
}

function startWorker(manifestUrl) {
  worker = new Worker("./worker.js", { type: "module" });
  worker.onmessage = (ev) => {
    const msg = ev.data;
    switch (msg.type) {
      case "progress":
        setProgress(msg.frac ?? 0, msg.label);
        break;
      case "ready":
        ready = true;
        hideProgress();
        setStatus(
          `Ready (${msg.backend || "webgpu"}, ${msg.quant || "?"}, ${manifestLabel})`
        );
        syncDescribeEnabled();
        break;
      case "token":
        caption.textContent += msg.text || "";
        break;
      case "done":
        busy = false;
        if (generatePulse) {
          clearInterval(generatePulse);
          generatePulse = null;
        }
        syncDescribeEnabled();
        setStatus(`Ready (webgpu, ${manifestLabel})`);
        break;
      case "error":
        busy = false;
        if (generatePulse) {
          clearInterval(generatePulse);
          generatePulse = null;
        }
        hideProgress();
        syncDescribeEnabled();
        setStatus(msg.message || "error");
        gate.textContent = msg.message || "error";
        gate.classList.remove("hidden");
        break;
      default:
        break;
    }
  };
  setProgress(0.02, "loading wasm…");
  worker.postMessage({ type: "init", manifestUrl });
}

function runDescribe() {
  busy = true;
  describeBtn.disabled = true;
  caption.textContent = "";
  setStatus("Generating… (first tokens can take a while)");
  if (generatePulse) clearInterval(generatePulse);
  let ticks = 0;
  generatePulse = setInterval(() => {
    ticks += 1;
    if (!busy) return;
    if (caption.textContent) {
      setStatus(`Generating… ${caption.textContent.length} chars`);
    } else {
      setStatus(`Generating… still working (${ticks * 2}s)`);
    }
  }, 2000);
  const temperature = Number.parseFloat(temperatureEl.value);
  let maxTokens = Number.parseInt(maxTokensEl.value, 10);
  if (!Number.isFinite(maxTokens)) maxTokens = 128;
  maxTokens = Math.max(16, Math.min(512, maxTokens));

  if (selectedVideo) {
    const transfers = [];
    const frameBytesList = selectedVideo.frameBytes.map((bytes) => {
      const transfer = bytes.buffer.slice(
        bytes.byteOffset,
        bytes.byteOffset + bytes.byteLength
      );
      transfers.push(transfer);
      return transfer;
    });
    worker.postMessage(
      {
        type: "describe_video",
        frameBytesList,
        timestamps: selectedVideo.timestamps || [],
        prompt: promptEl.value || "Describe this scene.",
        maxTokens,
        temperature: Number.isFinite(temperature) ? temperature : 0,
        probe: probeEnabled,
      },
      transfers
    );
    return;
  }

  const transfers = [];
  const imageBytesList = selectedImages.map((item) => {
    const transfer = item.bytes.buffer.slice(
      item.bytes.byteOffset,
      item.bytes.byteOffset + item.bytes.byteLength
    );
    transfers.push(transfer);
    return transfer;
  });
  worker.postMessage(
    {
      type: "describe",
      imageBytesList,
      prompt: promptEl.value || "Describe this scene.",
      maxTokens,
      temperature: Number.isFinite(temperature) ? temperature : 0,
      probe: probeEnabled,
    },
    transfers
  );
}

async function main() {
  bindUi();
  if (!requireWebGpu()) return;
  setStatus("Resolving weights…");
  const manifestUrl = await resolveManifestUrl();
  setStatus(
    manifestLabel === "local"
      ? "Starting worker (local weights)…"
      : "Starting worker…"
  );
  startWorker(manifestUrl);
}

main();
