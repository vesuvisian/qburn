/**
 * Web Worker: loads qburn wasm + weights, runs describe without blocking UI.
 *
 * Expects `./pkg/qburn_web.js` from wasm-pack (staged next to this file via
 * `bash web/stage.sh`). Messages from the main thread:
 *   { type: "init", manifestUrl }
 *   { type: "describe", imageBytes, prompt, maxTokens, temperature }
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

async function describe(imageBytes, prompt, maxTokens, temperature, probe) {
  if (!app) {
    postMessage({ type: "error", message: "model not ready" });
    return;
  }
  // Accept transferred ArrayBuffer or Uint8Array from the main thread.
  const bytes =
    imageBytes instanceof ArrayBuffer ? new Uint8Array(imageBytes) : imageBytes;
  const onToken = (piece) => {
    postMessage({ type: "token", text: piece });
  };
  const full = await app.describe(
    bytes,
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
        msg.imageBytes,
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
