#!/usr/bin/env node
/**
 * qburn-tjs — local transformers.js CLI (describe + prompt).
 * Comparison twin to `cargo run -p qburn …`; uses ONNX via @huggingface/transformers.
 */

import fs from "node:fs";
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
} from "@huggingface/transformers";

import {
  createTjsInfer,
  DEFAULT_DTYPE,
  DEFAULT_MODEL_ID,
  MAX_IMAGE_EDGE,
} from "../../web/static/tjs-infer.js";
import { parseArgs, usage } from "./lib/args.js";
import { configureWeightCache } from "./lib/cache.js";
import { classifyDescribePaths, loadVideoFramePaths } from "./lib/fs.js";

const infer = createTjsInfer({
  AutoProcessor,
  Qwen3VLForConditionalGeneration,
  RawImage,
  TextStreamer,
  EosTokenCriteria,
  InterruptableStoppingCriteria,
  StoppingCriteriaList,
  cat,
});

function isLocalModelPath(model) {
  if (!model) return false;
  // Hub ids look like "org/name"; local paths exist on disk.
  try {
    return fs.existsSync(model) && fs.statSync(model).isDirectory();
  } catch {
    return false;
  }
}

function configureEnv(model) {
  // Persist Hub downloads under weights/tjs-cache (not node_modules/.cache).
  const { cacheDir, migrated } = configureWeightCache(env);
  if (migrated) {
    console.error(`migrated ONNX cache → ${cacheDir}`);
  }
  if (isLocalModelPath(model)) {
    env.allowLocalModels = true;
    env.allowRemoteModels = false;
    // Point localModelPath at the parent so model id can be the dirname basename,
    // or pass an absolute path directly to from_pretrained.
  } else {
    env.allowLocalModels = false;
    env.allowRemoteModels = true;
  }
  return cacheDir;
}

function printMetrics({ loadMs, ttftMs, totalMs, tokens }) {
  const parts = [];
  if (loadMs != null) parts.push(`load=${loadMs.toFixed(1)}ms`);
  if (ttftMs != null) parts.push(`ttft=${ttftMs.toFixed(1)}ms`);
  else parts.push("ttft=n/a");
  parts.push(`total=${(totalMs / 1000).toFixed(2)}s`);
  parts.push(`tokens=${tokens}`);
  if (tokens > 1 && ttftMs != null && totalMs > ttftMs) {
    const decodeMs = totalMs - ttftMs;
    const tps = (tokens - 1) / (decodeMs / 1000);
    parts.push(`tok/s=${tps.toFixed(2)}`);
  }
  console.error(`metrics: ${parts.join("  ")}`);
}

async function loadModel(modelId, device, cacheDir) {
  const t0 = performance.now();
  console.error(`loading ${modelId} on ${device}…`);
  if (cacheDir) console.error(`cache: ${cacheDir}`);
  const { processor, model } = await infer.loadVlModel({
    modelId,
    device,
    dtype: DEFAULT_DTYPE,
    progress: (() => {
      let lastLabel = "";
      let lastPct = -1;
      /** Files that showed a partial progress tick (real download). */
      const downloading = new Set();
      return (data) => {
        if (!data || data.status !== "progress") return;
        const total = data.total || 0;
        const loaded = data.loaded || 0;
        if (total <= 0) return;
        const pct = Math.min(100, Math.round((100 * loaded) / total));
        const file = data.file
          ? String(data.file).split(/[/\\]/).pop()
          : "weights";
        if (pct < 100) downloading.add(file);
        // Cache hits often jump straight to 100%; label accordingly.
        const verb = downloading.has(file) ? "download" : "load";
        const label = `${verb} ${file}`;
        // Throttle: update when file changes or every 5% (incl. 100%).
        if (label !== lastLabel) {
          lastLabel = label;
          lastPct = -1;
        }
        if (pct >= lastPct + 5 || (pct === 100 && lastPct < 100)) {
          lastPct = pct;
          process.stderr.write(`\r${label} ${pct}%   `);
        }
      };
    })(),
  });
  process.stderr.write("\n");
  const loadMs = performance.now() - t0;
  console.error(`ready (${(loadMs / 1000).toFixed(1)}s)`);
  return { processor, model, loadMs };
}

async function loadImages(paths) {
  const out = [];
  for (const imgPath of paths) {
    let image = await RawImage.read(imgPath);
    image = await infer.resizeLongEdge(image, MAX_IMAGE_EDGE);
    out.push(image);
  }
  return out;
}

async function runDescribeOnce({
  processor,
  model,
  images,
  frames,
  timestamps,
  prompt,
  maxTokens,
  temperature,
  metrics,
  loadMs,
}) {
  let firstTokenAt = null;
  const genStart = performance.now();
  const onToken = (piece) => {
    if (firstTokenAt == null) firstTokenAt = performance.now();
    process.stdout.write(piece);
  };

  const result = frames
    ? await infer.describeVideo({
        processor,
        model,
        frames,
        timestamps,
        prompt,
        maxTokens,
        temperature,
        onToken,
      })
    : await infer.describeImage({
        processor,
        model,
        images,
        prompt,
        maxTokens,
        temperature,
        onToken,
      });
  const { text, newTokenCount } = result;
  const totalMs = performance.now() - genStart;
  if (!text.endsWith("\n")) process.stdout.write("\n");

  let ttftMs = null;
  if (metrics) {
    ttftMs = firstTokenAt != null ? firstTokenAt - genStart : null;
    printMetrics({
      loadMs,
      ttftMs,
      totalMs,
      tokens: newTokenCount,
    });
  }
  void text;
  return ttftMs;
}

async function runDescribeVideo(opts, source) {
  const modelId = opts.model || DEFAULT_MODEL_ID;
  const cacheDir = configureEnv(modelId);
  const { processor, model, loadMs } = await loadModel(
    modelId,
    opts.device,
    cacheDir
  );

  const label =
    source.kind === "frames" ? `${source.path} (frames)` : source.path;
  console.error(`video: ${label}`);
  const { paths: framePaths, timestamps } = loadVideoFramePaths(source, {
    fps: opts.fps,
    numFrames: opts.numFrames,
  });
  console.error(`sampled ${framePaths.length} frame(s)`);
  const frames = await loadImages(framePaths);
  await runDescribeOnce({
    processor,
    model,
    frames,
    timestamps,
    prompt: opts.prompt,
    maxTokens: opts.maxTokens,
    temperature: opts.temperature,
    metrics: opts.metrics,
    loadMs,
  });
}

async function runDescribe(opts) {
  const media = classifyDescribePaths(opts.paths, opts.recursive, opts.video);
  if (media.type === "video") {
    if (opts.together) {
      throw new Error("--together is image-only; omit it for video");
    }
    return runDescribeVideo(opts, media.source);
  }

  if (opts.fps != null || opts.numFrames != null) {
    throw new Error("--fps / --num-frames apply only to video describe");
  }

  const modelId = opts.model || DEFAULT_MODEL_ID;
  const cacheDir = configureEnv(modelId);
  const paths = media.paths;
  const { processor, model, loadMs } = await loadModel(
    modelId,
    opts.device,
    cacheDir
  );

  if (opts.together) {
    console.error(`multi-image prompt: ${paths.length} file(s)`);
    for (let i = 0; i < paths.length; i++) {
      console.error(`  [${i + 1}] ${paths[i]}`);
    }
    const images = await loadImages(paths);
    await runDescribeOnce({
      processor,
      model,
      images,
      prompt: opts.prompt,
      maxTokens: opts.maxTokens,
      temperature: opts.temperature,
      metrics: opts.metrics,
      loadMs,
    });
    return;
  }

  const ttfts = [];
  for (let i = 0; i < paths.length; i++) {
    const imgPath = paths[i];
    if (paths.length > 1) {
      console.error(`\n=== [${i + 1}/${paths.length}] ${imgPath}`);
    } else {
      console.error(`image: ${imgPath}`);
    }

    const images = await loadImages([imgPath]);
    const ttftMs = await runDescribeOnce({
      processor,
      model,
      images,
      prompt: opts.prompt,
      maxTokens: opts.maxTokens,
      temperature: opts.temperature,
      metrics: opts.metrics,
      loadMs: i === 0 ? loadMs : null,
    });
    if (ttftMs != null) ttfts.push(ttftMs);
  }

  if (opts.metrics && ttfts.length > 1) {
    const avg = ttfts.reduce((a, b) => a + b, 0) / ttfts.length;
    const min = Math.min(...ttfts);
    const max = Math.max(...ttfts);
    console.error(
      `batch ttft: avg=${avg.toFixed(1)}ms  min=${min.toFixed(1)}ms  max=${max.toFixed(1)}ms  n=${ttfts.length}`
    );
  }
}

async function runPrompt(opts) {
  const modelId = opts.model || DEFAULT_MODEL_ID;
  const cacheDir = configureEnv(modelId);
  const { processor, model, loadMs } = await loadModel(
    modelId,
    opts.device,
    cacheDir
  );

  let firstTokenAt = null;
  let streamed = false;
  const genStart = performance.now();
  const onToken =
    opts.stream || opts.metrics
      ? (piece) => {
          if (firstTokenAt == null) firstTokenAt = performance.now();
          if (opts.stream) {
            streamed = true;
            process.stdout.write(piece);
          }
        }
      : undefined;

  const { text, newTokenCount } = await infer.generatePrompt({
    processor,
    model,
    prompt: opts.userPrompt,
    system: opts.system,
    maxTokens: opts.maxTokens,
    temperature: opts.temperature,
    onToken,
  });
  const totalMs = performance.now() - genStart;

  if (opts.stream) {
    if (!streamed && text) process.stdout.write(text);
    if (!text.endsWith("\n")) process.stdout.write("\n");
  } else {
    process.stdout.write(text.endsWith("\n") ? text : `${text}\n`);
  }

  if (opts.metrics) {
    const ttftMs = firstTokenAt != null ? firstTokenAt - genStart : null;
    printMetrics({ loadMs, ttftMs, totalMs, tokens: newTokenCount });
  }
}

async function main() {
  let opts;
  try {
    opts = parseArgs(process.argv.slice(2));
  } catch (err) {
    console.error(err.message || err);
    process.exit(2);
  }

  if (opts.help) {
    process.stdout.write(usage());
    return;
  }

  try {
    if (opts.command === "describe") await runDescribe(opts);
    else await runPrompt(opts);
  } catch (err) {
    console.error(err.stack || err.message || err);
    process.exit(1);
  }
}

main();
