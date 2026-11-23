/**
 * Web Worker: transformers.js + ONNX Runtime WebGPU for Qwen3-VL.
 *
 * Messages from the main thread:
 *   { type: "init", modelId }
 *   { type: "describe", imageBytesList, prompt, maxTokens, temperature }
 *   { type: "describe_video", frameBytesList, prompt, maxTokens, temperature }
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
  cat,
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
  cat,
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

async function bytesToRawImages(byteList) {
  const images = [];
  for (const imageBytes of byteList) {
    const bytes =
      imageBytes instanceof ArrayBuffer ? new Uint8Array(imageBytes) : imageBytes;
    const blob = new Blob([bytes]);
    let image = await RawImage.fromBlob(blob);
    image = await infer.resizeLongEdge(image, MAX_IMAGE_EDGE);
    images.push(image);
  }
  return images;
}

async function describe(imageBytesList, prompt, maxTokens, temperature) {
  if (!processor || !model) {
    postMessage({ type: "error", message: "model not ready" });
    return;
  }

  const list = Array.isArray(imageBytesList) ? imageBytesList : [imageBytesList];
  if (list.length === 0) {
    postMessage({ type: "error", message: "describe requires at least one image" });
    return;
  }
  if (list.length > 4) {
    postMessage({
      type: "error",
      message: "browser demo supports at most 4 images per prompt",
    });
    return;
  }

  const images = await bytesToRawImages(list);
  const { text } = await infer.describeImage({
    processor,
    model,
    images,
    prompt,
    maxTokens,
    temperature,
    onToken: (piece) => postMessage({ type: "token", text: piece }),
  });
  postMessage({ type: "done", text });
}

async function describeVideo(
  frameBytesList,
  timestamps,
  prompt,
  maxTokens,
  temperature
) {
  if (!processor || !model) {
    postMessage({ type: "error", message: "model not ready" });
    return;
  }
  const list = Array.isArray(frameBytesList) ? frameBytesList : [frameBytesList];
  if (list.length < 4) {
    postMessage({
      type: "error",
      message: "describe_video requires at least 4 frames",
    });
    return;
  }
  if (list.length > 8) {
    postMessage({
      type: "error",
      message: "browser demo supports at most 8 video frames",
    });
    return;
  }

  const frames = await bytesToRawImages(list);
  const { text } = await infer.describeVideo({
    processor,
    model,
    frames,
    timestamps: Array.isArray(timestamps) ? timestamps : [],
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
        msg.imageBytesList ?? msg.imageBytes,
        msg.prompt,
        msg.maxTokens,
        msg.temperature
      );
    } else if (msg.type === "describe_video") {
      await describeVideo(
        msg.frameBytesList,
        msg.timestamps,
        msg.prompt,
        msg.maxTokens,
        msg.temperature
      );
    }
  } catch (err) {
    reportError(err);
  }
};
