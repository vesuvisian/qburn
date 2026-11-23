/**
 * qburn transformers.js demo — WebGPU gate, ONNX model load via worker, streaming caption.
 *
 * Model resolution (first match wins):
 *   1. `?model=` query param (Hugging Face Hub id)
 *   2. Default: onnx-community/Qwen3-VL-2B-Instruct-ONNX
 *
 * Media: up to 4 images **or** one video (exclusive).
 */

import {
  isImageFile,
  isVideoFile,
  sampleVideoFrames,
  VIDEO_MAX_FRAMES_WEB,
  VIDEO_MIN_FRAMES,
} from "./video-sample.js";

const DEFAULT_MODEL_ID = "onnx-community/Qwen3-VL-2B-Instruct-ONNX";
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
 *   url: string,
 *   revokePreview: () => void,
 * }}
 */
let selectedVideo = null;
let ready = false;
let busy = false;
let modelLabel = DEFAULT_MODEL_ID;
let generatePulse = null;

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

function resolveModelId() {
  const params = new URLSearchParams(location.search);
  const fromQuery = params.get("model");
  if (fromQuery && fromQuery.trim()) {
    modelLabel = fromQuery.trim();
    return modelLabel;
  }
  modelLabel = DEFAULT_MODEL_ID;
  return modelLabel;
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
    selectedVideo.revokePreview?.();
    selectedVideo = null;
  }
}

function renderThumbs() {
  thumbs.replaceChildren();
  if (selectedVideo) {
    const wrap = document.createElement("div");
    wrap.className = "thumb";
    const vid = document.createElement("video");
    vid.src = selectedVideo.url;
    vid.muted = true;
    vid.playsInline = true;
    vid.controls = true;
    vid.setAttribute("aria-label", "Selected video");
    const remove = document.createElement("button");
    remove.type = "button";
    remove.setAttribute("aria-label", "Remove video");
    remove.textContent = "×";
    remove.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      clearVideo();
      renderThumbs();
      caption.textContent = "";
    });
    wrap.append(vid, remove);
    thumbs.append(wrap);
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
      const sampled = await sampleVideoFrames(videos[0], {
        numFrames: VIDEO_MIN_FRAMES,
      });
      if (sampled.frameBytes.length < VIDEO_MIN_FRAMES) {
        sampled.revokePreview?.();
        setStatus(`Need at least ${VIDEO_MIN_FRAMES} frames from the video`);
        return;
      }
      selectedVideo = {
        kind: "video",
        frameBytes: sampled.frameBytes.map((ab) => new Uint8Array(ab)),
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
  if (selectedVideo) clearVideo();
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

function startWorker(modelId) {
  worker = new Worker("./tjs-worker.js", { type: "module" });
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
          `Ready (${msg.backend || "webgpu"}, ${msg.quant || "q4f16"}, transformers.js)`
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
        if (typeof msg.text === "string") {
          caption.textContent = msg.text;
        }
        syncDescribeEnabled();
        setStatus("Ready (webgpu, transformers.js)");
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
  setProgress(0.02, "starting worker…");
  worker.postMessage({ type: "init", modelId });
}

function runDescribe() {
  busy = true;
  describeBtn.disabled = true;
  caption.textContent = "";
  setStatus("Generating…");
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
    },
    transfers
  );
}

async function main() {
  bindUi();
  if (!requireWebGpu()) return;
  const modelId = resolveModelId();
  setStatus(`Loading ${modelId}…`);
  startWorker(modelId);
}

main();
