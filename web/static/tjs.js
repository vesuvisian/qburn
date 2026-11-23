/**
 * qburn transformers.js demo — WebGPU gate, ONNX model load via worker, streaming caption.
 *
 * Model resolution (first match wins):
 *   1. `?model=` query param (Hugging Face Hub id)
 *   2. Default: onnx-community/Qwen3-VL-2B-Instruct-ONNX
 */

const DEFAULT_MODEL_ID = "onnx-community/Qwen3-VL-2B-Instruct-ONNX";

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
        if (typeof msg.text === "string") {
          caption.textContent = msg.text;
        }
        describeBtn.disabled = !imageBytes;
        setStatus("Ready (webgpu, transformers.js)");
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
    },
    [transfer]
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
