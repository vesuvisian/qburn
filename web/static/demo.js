/**
 * qburn browser demo — WebGPU gate, weight load via worker, streaming caption.
 *
 * Manifest resolution (first match wins):
 *   1. `?manifest=` query param
 *   2. `window.QBURN_MANIFEST` (set before this script)
 *   3. `./weights/manifest.json` when served locally via `preview.py --weights`
 *   4. Hugging Face Hub default
 *
 * See architecture.html and the repo README for convert / host steps.
 */

const HUB_MANIFEST =
  "https://huggingface.co/vesuvisian/qburn-weights/resolve/main/manifest.json";

const gate = document.getElementById("gate");
const statusEl = document.getElementById("status");
const progress = document.getElementById("progress");
const barFill = document.getElementById("barFill");
const progressLabel = document.getElementById("progressLabel");
const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const promptEl = document.getElementById("prompt");
const maxTokensEl = document.getElementById("maxTokens");
const temperatureEl = document.getElementById("temperature");
const describeBtn = document.getElementById("describeBtn");
const caption = document.getElementById("caption");

let worker = null;
let imageBytes = null;
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
    // Some static servers dislike HEAD; try a tiny GET.
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

function bindUi() {
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag"));
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag");
    const file = e.dataTransfer?.files?.[0];
    if (file) ingestFile(file);
  });
  fileInput.addEventListener("change", () => {
    const file = fileInput.files?.[0];
    if (file) ingestFile(file);
  });
  describeBtn.addEventListener("click", () => {
    if (!ready || !imageBytes || busy) return;
    runDescribe();
  });
}

async function ingestFile(file) {
  imageBytes = new Uint8Array(await file.arrayBuffer());
  preview.src = URL.createObjectURL(file);
  preview.classList.remove("hidden");
  describeBtn.disabled = !ready;
  caption.textContent = "";
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
        describeBtn.disabled = !imageBytes;
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
        describeBtn.disabled = !imageBytes;
        setStatus(`Ready (webgpu, ${manifestLabel})`);
        break;
      case "error":
        busy = false;
        if (generatePulse) {
          clearInterval(generatePulse);
          generatePulse = null;
        }
        hideProgress();
        describeBtn.disabled = !imageBytes;
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
  // Copy + transfer the ArrayBuffer so the worker avoids a structured-clone
  // of the full upload, while keeping `imageBytes` for re-runs.
  const transfer = imageBytes.buffer.slice(
    imageBytes.byteOffset,
    imageBytes.byteOffset + imageBytes.byteLength
  );
  worker.postMessage(
    {
      type: "describe",
      imageBytes: transfer,
      prompt: promptEl.value || "Describe this scene.",
      maxTokens,
      temperature: Number.isFinite(temperature) ? temperature : 0,
      probe: probeEnabled,
    },
    [transfer]
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
