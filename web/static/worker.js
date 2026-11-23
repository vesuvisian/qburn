/**
 * Web Worker: loads qburn wasm + weights, runs describe without blocking UI.
 *
 * Expects `./pkg/qburn_web.js` from wasm-pack (staged next to this file via
 * `bash web/stage.sh`). Messages from the main thread:
 *   { type: "init", manifestUrl }
 *   { type: "describe", imageBytesList, prompt, maxTokens, temperature }
 *   { type: "describe_video", frameBytesList, timestamps, prompt, maxTokens, temperature }
 * Replies: progress | ready | token | done | error
 */

let app = null;

function reportError(err) {
  const message =
    (err && err.message) ||
    (typeof err === "string" ? err : null) ||
    String(err || "worker error");
  postMessage({ type: "error", message });
}

self.addEventListener("error", (ev) => {
  reportError(ev.error || ev.message || "worker error");
});
self.addEventListener("unhandledrejection", (ev) => {
  reportError(ev.reason || "unhandledrejection");
});

async function init(manifestUrl) {
  postMessage({ type: "progress", frac: 0.05, label: "import wasm…" });
  const mod = await import("./pkg/qburn_web.js");
  await mod.default();
  postMessage({ type: "progress", frac: 0.15, label: "fetch weights…" });

  // Progress is coarse until streaming fetch lands in Rust; pulse while awaiting.
  const pulse = setInterval(() => {
    postMessage({ type: "progress", frac: 0.35, label: "downloading / caching weights…" });
  }, 800);

  try {
    app = await mod.QburnApp.init(manifestUrl);
  } finally {
    clearInterval(pulse);
  }

  postMessage({
    type: "ready",
    backend: app.backendName(),
    quant: app.quantName(),
  });
}

function toUint8List(list) {
  const arr = Array.isArray(list) ? list : [list];
  return arr.map((imageBytes) =>
    imageBytes instanceof ArrayBuffer ? new Uint8Array(imageBytes) : imageBytes
  );
}

async function describe(imageBytesList, prompt, maxTokens, temperature, probe) {
  if (!app) {
    postMessage({ type: "error", message: "model not ready" });
    return;
  }
  const images = toUint8List(imageBytesList);
  const onToken = (piece) => {
    postMessage({ type: "token", text: piece });
  };
  const full = await app.describe(
    images,
    prompt,
    maxTokens,
    temperature,
    !!probe,
    onToken
  );
  postMessage({ type: "done", text: full });
}

async function describeVideo(
  frameBytesList,
  timestamps,
  prompt,
  maxTokens,
  temperature,
  probe
) {
  if (!app) {
    postMessage({ type: "error", message: "model not ready" });
    return;
  }
  const frames = toUint8List(frameBytesList);
  const times = Float64Array.from(
    Array.isArray(timestamps) ? timestamps : []
  );
  const onToken = (piece) => {
    postMessage({ type: "token", text: piece });
  };
  const full = await app.describeVideo(
    frames,
    times,
    prompt,
    maxTokens,
    temperature,
    !!probe,
    onToken
  );
  postMessage({ type: "done", text: full });
}

onmessage = async (ev) => {
  const msg = ev.data;
  try {
    if (msg.type === "init") {
      await init(msg.manifestUrl);
    } else if (msg.type === "describe") {
      await describe(
        msg.imageBytesList ?? msg.imageBytes,
        msg.prompt,
        msg.maxTokens ?? 128,
        msg.temperature ?? 0,
        msg.probe ?? false
      );
    } else if (msg.type === "describe_video") {
      await describeVideo(
        msg.frameBytesList,
        msg.timestamps,
        msg.prompt,
        msg.maxTokens ?? 128,
        msg.temperature ?? 0,
        msg.probe ?? false
      );
    }
  } catch (err) {
    reportError(err);
  }
};
