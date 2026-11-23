/**
 * Web Worker: transformers.js + ONNX Runtime WebGPU for Qwen3-VL.
 *
 * Messages from the main thread:
 *   { type: "init", modelId }
 *   { type: "describe", imageBytes, prompt, maxTokens, temperature }
 * Replies: progress | ready | token | done | error
 */

import {
  AutoProcessor,
  Qwen3VLForConditionalGeneration,
  RawImage,
  TextStreamer,
  EosTokenCriteria,
  InterruptableStoppingCriteria,
  StoppingCriteriaList,
  env,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2";

import {
  createTjsInfer,
  DEFAULT_DTYPE,
  MAX_IMAGE_EDGE,
} from "./tjs-infer.js";

const infer = createTjsInfer({
  AutoProcessor,
  Qwen3VLForConditionalGeneration,
  TextStreamer,
  EosTokenCriteria,
  InterruptableStoppingCriteria,
  StoppingCriteriaList,
});

let processor = null;
let model = null;
let modelId = null;

function reportError(err) {
  const message =
    (err && err.message) ||
    (typeof err === "string" ? err : null) ||
    String(err || "worker error");
  const stack = err && err.stack ? String(err.stack) : "";
  postMessage({ type: "error", message: stack ? `${message}\n${stack}` : message });
}

self.addEventListener("error", (ev) => {
  reportError(ev.error || ev.message || "worker error");
});
self.addEventListener("unhandledrejection", (ev) => {
  reportError(ev.reason || "unhandledrejection");
});

async function init(id) {
  modelId = id;
  // Cache models in the browser; allow remote Hub fetches.
  env.allowLocalModels = false;
  env.useBrowserCache = true;

  postMessage({ type: "progress", frac: 0.05, label: "import transformers.js…" });
  postMessage({ type: "progress", frac: 0.15, label: "load processor…" });

  const loaded = await infer.loadVlModel({
    modelId,
    device: "webgpu",
    dtype: DEFAULT_DTYPE,
    progress: (data) => {
      if (!data || data.status !== "progress") return;
      const total = data.total || 0;
      const loadedBytes = data.loaded || 0;
      if (total > 0) {
        const frac = 0.3 + 0.65 * Math.min(1, loadedBytes / total);
        const file = data.file ? String(data.file).split("/").pop() : "weights";
        postMessage({
          type: "progress",
          frac,
          label: `download ${file}…`,
        });
      }
    },
  });
  processor = loaded.processor;
  model = loaded.model;

  postMessage({
    type: "ready",
    backend: "webgpu",
    quant: "q4f16",
    modelId,
  });
}

async function describe(imageBytes, prompt, maxTokens, temperature) {
  if (!processor || !model) {
    postMessage({ type: "error", message: "model not ready" });
    return;
  }

  const bytes =
    imageBytes instanceof ArrayBuffer ? new Uint8Array(imageBytes) : imageBytes;
  const blob = new Blob([bytes]);
  let image = await RawImage.fromBlob(blob);
  image = await infer.resizeLongEdge(image, MAX_IMAGE_EDGE);

  const { text } = await infer.describeImage({
    processor,
    model,
    image,
    prompt,
    maxTokens,
    temperature,
    onToken: (piece) => postMessage({ type: "token", text: piece }),
  });
  postMessage({ type: "done", text });
}

onmessage = async (ev) => {
  const msg = ev.data;
  try {
    if (msg.type === "init") {
      await init(msg.modelId);
    } else if (msg.type === "describe") {
      await describe(
        msg.imageBytes,
        msg.prompt,
        msg.maxTokens,
        msg.temperature
      );
    }
  } catch (err) {
    reportError(err);
  }
};
