/**
 * Shared transformers.js helpers for the browser worker and Node CLI.
 *
 * Import transformers bindings in the host, then call createTjsInfer(api).
 * That keeps CDN vs npm resolution out of this file.
 */

/** Long-edge resize so vision memory stays in a browser-friendly band. */
export const MAX_IMAGE_EDGE = 448;

/**
 * HF Qwen3-VL video size budget (`shortest_edge` / `longest_edge` on the
 * video processor). Applied to `t_bar * h_bar * w_bar` so token cost scales
 * with clip length instead of spending a full image budget per frame.
 */
export const VIDEO_MIN_PIXELS = 128 * 32 * 32; // 131_072
export const VIDEO_MAX_PIXELS = 32 * 32 * 768; // 786_432
export const VIDEO_SPATIAL_FACTOR = 32; // patch_size 16 × merge_size 2

export const DEFAULT_MODEL_ID = "onnx-community/Qwen3-VL-2B-Instruct-ONNX";

/** Default dtype mix matching the WebGPU demo. */
export const DEFAULT_DTYPE = Object.freeze({
  embed_tokens: "fp16",
  vision_encoder: "fp16",
  decoder_model_merged: "q4f16",
});

/**
 * @param {object} api
 * @param {typeof import("@huggingface/transformers").AutoProcessor} api.AutoProcessor
 * @param {typeof import("@huggingface/transformers").Qwen3VLForConditionalGeneration} api.Qwen3VLForConditionalGeneration
 * @param {typeof import("@huggingface/transformers").TextStreamer} api.TextStreamer
 * @param {typeof import("@huggingface/transformers").EosTokenCriteria} api.EosTokenCriteria
 * @param {typeof import("@huggingface/transformers").InterruptableStoppingCriteria} api.InterruptableStoppingCriteria
 * @param {typeof import("@huggingface/transformers").StoppingCriteriaList} api.StoppingCriteriaList
 * @param {typeof import("@huggingface/transformers").RawImage} [api.RawImage] — required for generatePrompt
 * @param {typeof import("@huggingface/transformers").cat} [api.cat] — required for multi-image (per-image concat)
 */
export function createTjsInfer(api) {
  const {
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    TextStreamer,
    EosTokenCriteria,
    InterruptableStoppingCriteria,
    StoppingCriteriaList,
    RawImage,
    cat,
  } = api;

  /** Qwen chat EOS ids: <|im_end|> and <|endoftext|> (generation_config.json). */
  function resolveEosTokenIds(tokenizer) {
    const ids = new Set();
    if (tokenizer?.eos_token_id != null) ids.add(Number(tokenizer.eos_token_id));
    if (tokenizer?.pad_token_id != null) ids.add(Number(tokenizer.pad_token_id));
    for (const tok of ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]) {
      const id = tokenizer?.token_to_id?.(tok);
      if (id != null) ids.add(Number(id));
    }
    // Fallbacks if the tokenizer map is incomplete in-browser.
    ids.add(151645); // <|im_end|>
    ids.add(151643); // <|endoftext|>
    ids.add(151644); // <|im_start|> (stop if the model starts another turn)
    return [...ids];
  }

  /**
   * Port of qburn::with_length_budget — soft cue so greedy decode aims to
   * finish before max_tokens cuts off.
   */
  function withLengthBudget(prompt, maxTokens) {
    let hint;
    if (maxTokens <= 64) hint = "Answer in one short sentence and stop.";
    else if (maxTokens <= 128)
      hint = "Keep the full answer to a few short sentences and stop when done.";
    else if (maxTokens <= 256)
      hint = "Keep the answer to one short paragraph and stop when done.";
    else hint = "Keep the answer concise and stop when done.";

    const lower = prompt.toLowerCase();
    if (
      lower.includes("short sentence") ||
      lower.includes("few sentences") ||
      lower.includes("briefly") ||
      lower.includes("concise") ||
      lower.includes("and stop")
    ) {
      return prompt;
    }
    return `${prompt} ${hint}`;
  }

  async function resizeLongEdge(image, maxEdge = MAX_IMAGE_EDGE) {
    const w = image.width;
    const h = image.height;
    const long = Math.max(w, h);
    if (long <= maxEdge) return image;
    const scale = maxEdge / long;
    // RawImage.resize is async (returns Promise<RawImage>).
    return await image.resize(Math.round(w * scale), Math.round(h * scale));
  }

  /**
   * HF Qwen3VLVideoProcessor.smart_resize — budget is over `t × h × w`, not
   * per frame, so longer clips shrink spatially instead of exploding tokens.
   */
  function videoSmartResize(
    numFrames,
    height,
    width,
    {
      temporalFactor = 2,
      factor = VIDEO_SPATIAL_FACTOR,
      minPixels = VIDEO_MIN_PIXELS,
      maxPixels = VIDEO_MAX_PIXELS,
    } = {}
  ) {
    let h = height;
    let w = width;
    if (h < factor || w < factor) {
      const scale = Math.max(factor / h, factor / w);
      h = Math.floor(h * scale);
      w = Math.floor(w * scale);
    }
    let hBar = Math.round(h / factor) * factor;
    let wBar = Math.round(w / factor) * factor;
    const tBar = Math.round(numFrames / temporalFactor) * temporalFactor;
    if (tBar * hBar * wBar > maxPixels) {
      const beta = Math.sqrt((numFrames * height * width) / maxPixels);
      hBar = Math.max(factor, Math.floor(height / beta / factor) * factor);
      wBar = Math.max(factor, Math.floor(width / beta / factor) * factor);
    } else if (tBar * hBar * wBar < minPixels) {
      const beta = Math.sqrt(minPixels / (numFrames * height * width));
      hBar = Math.ceil((height * beta) / factor) * factor;
      wBar = Math.ceil((width * beta) / factor) * factor;
    }
    return { height: hBar, width: wBar };
  }

  /**
   * @param {object} opts
   * @param {string} opts.modelId
   * @param {string} [opts.device]
   * @param {object} [opts.dtype]
   * @param {(data: object) => void} [opts.progress]
   */
  async function loadVlModel({
    modelId,
    device = "webgpu",
    dtype = DEFAULT_DTYPE,
    progress,
  }) {
    const processor = await AutoProcessor.from_pretrained(modelId);
    const model = await Qwen3VLForConditionalGeneration.from_pretrained(modelId, {
      device,
      dtype,
      progress_callback: progress,
    });

    // Ensure Hub generation_config stop ids stick on the loaded model.
    const eosIds = resolveEosTokenIds(processor.tokenizer);
    const genCfg = model.configs?.generation_config;
    if (genCfg && typeof genCfg === "object") {
      genCfg.eos_token_id = eosIds;
      if (genCfg.pad_token_id == null) genCfg.pad_token_id = 151643;
    }

    return { processor, model };
  }

  /**
   * Detect greedy sentence-loop collapse: the same long phrase appears 3+ times.
   * Used to interrupt generate when EOS never arrives.
   */
  function detectRepeatingCollapse(text) {
    if (text.length < 120) return false;
    // Prefer sentence-sized chunks; fall back to a trailing window.
    const sentences = text.split(/(?<=[.!?])\s+/).filter((s) => s.length >= 40);
    if (sentences.length >= 3) {
      const last = sentences[sentences.length - 1];
      let hits = 0;
      for (const s of sentences) {
        if (s === last) hits += 1;
      }
      if (hits >= 3) return true;
    }
    const win = Math.min(80, Math.floor(text.length / 3));
    if (win < 40) return false;
    const needle = text.slice(-win);
    let from = 0;
    let hits = 0;
    while (from <= text.length - win) {
      const at = text.indexOf(needle, from);
      if (at < 0) break;
      hits += 1;
      if (hits >= 3) return true;
      from = at + win;
    }
    return false;
  }

  /** Drop trailing looped sentences once collapse is detected. */
  function trimCollapsedRepetition(text) {
    if (!detectRepeatingCollapse(text)) return text;
    const sentences = text.split(/(?<=[.!?])\s+/);
    if (sentences.length < 3) return text;
    const seen = new Map();
    const kept = [];
    for (const s of sentences) {
      if (s.length >= 40) {
        const n = (seen.get(s) || 0) + 1;
        seen.set(s, n);
        if (n >= 2) break; // keep first occurrence only
      }
      kept.push(s);
    }
    return kept.join(" ").trim();
  }

  /**
   * Run generate with streaming + EOS interrupt; return decoded new tokens.
   * @param {object} opts
   * @param {*} opts.processor
   * @param {*} opts.model
   * @param {*} opts.inputs — processor outputs (input_ids, …)
   * @param {number} opts.maxTokens
   * @param {number} opts.temperature
   * @param {(piece: string) => void} [opts.onToken]
   */
  async function generateFromInputs({
    processor,
    model,
    inputs,
    maxTokens,
    temperature,
    onToken,
  }) {
    const budget = maxTokens || 128;
    const temp = Number(temperature);
    const doSample = Number.isFinite(temp) && temp > 0;
    const eos_token_id = resolveEosTokenIds(processor.tokenizer);
    const eosSet = new Set(eos_token_id);
    const pad_token_id =
      processor.tokenizer.pad_token_id != null
        ? Number(processor.tokenizer.pad_token_id)
        : 151643;

    // Belt-and-suspenders: built-in EosTokenCriteria often misses BigInt/number
    // mismatches; interrupt as soon as a stop id is sampled.
    const interrupt = new InterruptableStoppingCriteria();
    const stopping_criteria = new StoppingCriteriaList();
    stopping_criteria.push(new EosTokenCriteria(eos_token_id));
    stopping_criteria.push(interrupt);

    let sawEos = false;
    let streamed = "";
    const streamer = new TextStreamer(processor.tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      // Do not gate on sawEos: token_callback may mark EOS in the same step
      // as the final text piece, and dropping it leaves short replies blank.
      callback_function: (piece) => {
        if (!piece) return;
        streamed += piece;
        // Greedy video decode can loop a sentence forever without emitting EOS.
        if (detectRepeatingCollapse(streamed)) {
          interrupt.interrupt();
          return;
        }
        if (onToken) onToken(piece);
      },
      token_callback_function: (tokens) => {
        if (!tokens || sawEos) return;
        for (const t of tokens) {
          if (eosSet.has(Number(t))) {
            sawEos = true;
            interrupt.interrupt();
            return;
          }
        }
      },
    });

    const genOpts = {
      ...inputs,
      generation_config: {
        max_new_tokens: budget,
        do_sample: doSample,
        eos_token_id,
        pad_token_id,
        // Greedy video decode is prone to sentence loops without EOS.
        repetition_penalty: doSample ? 1.0 : 1.1,
        no_repeat_ngram_size: doSample ? 0 : 5,
        ...(doSample ? { temperature: temp } : {}),
      },
      eos_token_id,
      pad_token_id,
      stopping_criteria,
      streamer,
    };

    const outputs = await model.generate(genOpts);
    const promptLen = inputs.input_ids.dims.at(-1);
    const newTokens = outputs.slice(null, [promptLen, null]);
    // Drop everything from the first EOS onward (if stopping lagged by a step).
    let tokenRows = newTokens.tolist();
    let full;
    let newTokenCount = 0;
    if (Array.isArray(tokenRows?.[0])) {
      const row = tokenRows[0];
      const cut = row.findIndex((t) => eosSet.has(Number(t)));
      const kept = cut >= 0 ? row.slice(0, cut) : row;
      newTokenCount = kept.length;
      if (cut >= 0) {
        full = processor.tokenizer
          .decode(kept, { skip_special_tokens: true })
          .trim();
      }
    }
    if (full == null) {
      const decoded = processor.batch_decode(newTokens, {
        skip_special_tokens: true,
      });
      full = (decoded[0] || "").trim();
      if (!newTokenCount && Array.isArray(tokenRows?.[0])) {
        newTokenCount = tokenRows[0].length;
      }
    }
    full = trimCollapsedRepetition(full);
    return { text: full, newTokenCount };
  }

  /**
   * Build VL model inputs for one or more images.
   *
   * transformers.js `Qwen2VLImageProcessor` treats a RawImage[] as one batch
   * and returns a single `image_grid_thw` row — that breaks multi-image pad
   * expansion. Process each image alone, then concat (same idea as HF Lfm3).
   *
   * @param {*} processor
   * @param {string} text — chat template with one `<|image_pad|>` per image
   * @param {Array} imageList — RawImage[]
   */
  async function buildVlInputs(processor, text, imageList) {
    if (imageList.length === 1) {
      return processor(text, imageList[0]);
    }
    if (typeof cat !== "function") {
      throw new Error(
        "multi-image describe requires createTjsInfer({ cat }) from @huggingface/transformers"
      );
    }

    const parts = await Promise.all(
      imageList.map((img) => processor.image_processor(img))
    );
    const pixel_values = cat(
      parts.map((p) => p.pixel_values),
      0
    );
    const image_grid_thw = cat(
      parts.map((p) => p.image_grid_thw),
      0
    );

    // Mirror Qwen2VLProcessor._call pad expansion over per-image grids.
    const merge_length = processor.image_processor.config.merge_size ** 2;
    const image_token = "<|image_pad|>";
    const grids = image_grid_thw.tolist();
    let index = 0;
    let expanded = text;
    while (expanded.includes(image_token)) {
      if (index >= grids.length) {
        throw new Error(
          `image pad count exceeds image_grid_thw rows (${grids.length})`
        );
      }
      const prod = Number(grids[index++].reduce((a, b) => a * b, 1n));
      expanded = expanded.replace(
        image_token,
        "<|placeholder|>".repeat(Math.floor(prod / merge_length))
      );
    }
    if (index !== grids.length) {
      throw new Error(
        `unused image_grid_thw rows: used ${index} of ${grids.length}`
      );
    }
    expanded = expanded.replaceAll("<|placeholder|>", image_token);
    const text_inputs = processor.tokenizer(expanded);
    return { ...text_inputs, pixel_values, image_grid_thw };
  }

  /**
   * Average frame timestamps within each temporal patch (HF Qwen3-VL).
   * @param {number[]} frameTimes
   * @param {number} temporalPatchSize
   */
  function temporalGroupTimestamps(frameTimes, temporalPatchSize = 2) {
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

  /**
   * Build VL model inputs for a single video (Qwen3-VL timestamped path).
   *
   * transformers.js `Qwen2VLImageProcessor` treats a RawImage[] batch as a
   * temporal stack when `dims[0] > 1`. Qwen3-VL expects the prompt to insert
   * `<X.X seconds>` + a vision block per temporal group, and RoPE to see
   * `video_grid_thw` rows with `t=1` (not one contiguous grid_t>1 span).
   *
   * @param {*} processor
   * @param {string} userText — user prompt (no vision placeholders)
   * @param {Array} frameList — RawImage[]
   * @param {number[]} [frameTimestamps] — seek times in seconds
   */
  async function buildVlVideoInputs(
    processor,
    userText,
    frameList,
    frameTimestamps = []
  ) {
    if (!frameList.length) {
      throw new Error("video requires at least one frame");
    }
    const tps = Number(processor.image_processor?.config?.temporal_patch_size) || 2;
    const mergeSize = Number(processor.image_processor?.config?.merge_size) || 2;
    const frames = frameList.slice();
    while (frames.length % tps !== 0) {
      frames.push(frames[frames.length - 1]);
    }

    let times =
      Array.isArray(frameTimestamps) && frameTimestamps.length === frameList.length
        ? frameTimestamps.slice()
        : frameList.map((_, i) => i / 2);
    while (times.length < frames.length) {
      times.push(times[times.length - 1]);
    }
    times = times.slice(0, frames.length);

    // Shared video spatial size (HF total-pixel budget). Without this, N frames
    // each keep a full image resolution → thousands of vision tokens and
    // greedy decode collapses into sentence loops.
    const { height: vh, width: vw } = videoSmartResize(
      frames.length,
      frames[0].height,
      frames[0].width
    );
    const sized = [];
    for (const frame of frames) {
      sized.push(
        frame.width === vw && frame.height === vh
          ? frame
          : await frame.resize(vw, vh)
      );
    }

    const ip = processor.image_processor;
    const prevMin = ip.min_pixels;
    const prevMax = ip.max_pixels;
    // Pin processor resize to the video budget so it does not re-inflate.
    ip.min_pixels = Math.min(vh * vw, VIDEO_MIN_PIXELS);
    ip.max_pixels = Math.max(vh * vw, vh * vw);
    let vision;
    try {
      vision = await ip(sized);
    } finally {
      ip.min_pixels = prevMin;
      ip.max_pixels = prevMax;
    }

    const grids = vision.image_grid_thw.tolist();
    if (grids.length !== 1) {
      throw new Error(`expected one video grid row, got ${grids.length}`);
    }
    const gt = Number(grids[0][0]);
    const gh = Number(grids[0][1]);
    const gw = Number(grids[0][2]);
    const tokensPerGroup = Math.floor((gh * gw) / (mergeSize * mergeSize));
    const groupTimes = temporalGroupTimestamps(times, tps);
    if (groupTimes.length !== gt) {
      throw new Error(
        `timestamp groups ${groupTimes.length} != video grid_t ${gt}`
      );
    }

    const visionStart = "<|vision_start|>";
    const visionEnd = "<|vision_end|>";
    const videoToken = "<|video_pad|>";
    let visionBlocks = "";
    for (const ts of groupTimes) {
      visionBlocks += `<${ts.toFixed(1)} seconds>`;
      visionBlocks += visionStart + videoToken.repeat(tokensPerGroup) + visionEnd;
    }
    const expanded =
      `<|im_start|>user\n${visionBlocks}${userText}<|im_end|>\n` +
      `<|im_start|>assistant\n`;
    const text_inputs = processor.tokenizer(expanded);

    // Vision encode keeps the full temporal grid; RoPE gets split t=1 rows
    // (Qwen3-VL get_rope_index expands video_grid_thw the same way).
    const TensorCtor = vision.image_grid_thw.constructor;
    const split = [];
    for (let i = 0; i < gt; i++) {
      split.push(1n, BigInt(gh), BigInt(gw));
    }
    const video_grid_thw = new TensorCtor("int64", BigInt64Array.from(split), [
      gt,
      3,
    ]);

    return {
      ...text_inputs,
      pixel_values: vision.pixel_values,
      image_grid_thw: vision.image_grid_thw,
      video_grid_thw,
    };
  }

  /**
   * Run generate while splicing vision features onto `<|video_pad|>` tokens.
   *
   * transformers.js gaps we paper over for a single video:
   * 1. Merge only looks at `image_token_id` — override to `video_token_id`.
   * 2. `forward_params` omits `video_grid_thw`, so `_prepare_model_inputs`
   *    drops it before `get_rope_index` — temporarily allow the key (ONNX
   *    session still picks only its own input names).
   */
  async function generateVideoFromInputs(opts) {
    const { model } = opts;
    const videoTokenId = model.config?.video_token_id;
    if (videoTokenId == null) {
      throw new Error("model.config.video_token_id missing");
    }
    const origMerge = model._merge_input_ids_with_image_features;
    model._merge_input_ids_with_image_features = function mergeVideoPads(kwargs) {
      return origMerge.call(this, {
        ...kwargs,
        image_token_id: videoTokenId,
      });
    };

    const params = model.forward_params;
    const hadVideoGrid = Array.isArray(params) && params.includes("video_grid_thw");
    if (Array.isArray(params) && !hadVideoGrid) {
      params.push("video_grid_thw");
    }

    try {
      return await generateFromInputs(opts);
    } finally {
      model._merge_input_ids_with_image_features = origMerge;
      if (Array.isArray(params) && !hadVideoGrid) {
        const i = params.indexOf("video_grid_thw");
        if (i >= 0) params.splice(i, 1);
      }
    }
  }

  /**
   * @param {object} opts
   * @param {*} opts.processor
   * @param {*} opts.model
   * @param {*} [opts.image] — single RawImage (legacy)
   * @param {*|Array} [opts.images] — RawImage or array of RawImage
   * @param {string} [opts.prompt]
   * @param {number} [opts.maxTokens]
   * @param {number} [opts.temperature]
   * @param {(piece: string) => void} [opts.onToken]
   */
  async function describeImage({
    processor,
    model,
    image,
    images,
    prompt,
    maxTokens,
    temperature,
    onToken,
  }) {
    const budget = maxTokens || 128;
    const userText = withLengthBudget(prompt || "Describe this scene.", budget);

    let imageList;
    if (images != null) {
      imageList = Array.isArray(images) ? images : [images];
    } else if (image != null) {
      imageList = [image];
    } else {
      throw new Error("describeImage requires image or images");
    }
    if (imageList.length === 0) {
      throw new Error("describeImage requires at least one image");
    }

    const content = [
      ...imageList.map(() => ({ type: "image" })),
      { type: "text", text: userText },
    ];
    const conversation = [{ role: "user", content }];
    const text = processor.apply_chat_template(conversation, {
      add_generation_prompt: true,
    });
    const inputs = await buildVlInputs(processor, text, imageList);
    return generateFromInputs({
      processor,
      model,
      inputs,
      maxTokens: budget,
      temperature,
      onToken,
    });
  }

  /**
   * Single-video describe with Qwen3-VL timestamped `<|video_pad|>` blocks.
   *
   * @param {object} opts
   * @param {*} opts.processor
   * @param {*} opts.model
   * @param {Array} opts.frames — RawImage[] sampled frames
   * @param {number[]} [opts.timestamps] — seek times in seconds
   * @param {string} [opts.prompt]
   * @param {number} [opts.maxTokens]
   * @param {number} [opts.temperature]
   * @param {(piece: string) => void} [opts.onToken]
   */
  async function describeVideo({
    processor,
    model,
    frames,
    timestamps,
    prompt,
    maxTokens,
    temperature,
    onToken,
  }) {
    const budget = maxTokens || 128;
    const userText = withLengthBudget(prompt || "Describe this scene.", budget);
    const frameList = Array.isArray(frames) ? frames : [frames];
    if (!frameList.length) {
      throw new Error("describeVideo requires frames");
    }

    const inputs = await buildVlVideoInputs(
      processor,
      userText,
      frameList,
      timestamps
    );
    return generateVideoFromInputs({
      processor,
      model,
      inputs,
      maxTokens: budget,
      temperature,
      onToken,
    });
  }

  /**
   * Text-oriented chat generation (CLI `prompt`).
   *
   * The Qwen3-VL ONNX graph expects vision inputs; a tiny neutral placeholder
   * image is attached so Node/ORT does not fall into the broken no-pixel path.
   * System text is prepended into the user turn (same idea as Burn's chat
   * template packing) because the VL processor path is image+user oriented.
   *
   * @param {object} opts
   * @param {*} opts.processor
   * @param {*} opts.model
   * @param {string} opts.prompt
   * @param {string} [opts.system]
   * @param {number} [opts.maxTokens]
   * @param {number} [opts.temperature]
   * @param {(piece: string) => void} [opts.onToken]
   */
  async function generatePrompt({
    processor,
    model,
    prompt,
    system = "You are a helpful assistant.",
    maxTokens,
    temperature,
    onToken,
  }) {
    if (!RawImage) {
      throw new Error("createTjsInfer: RawImage is required for generatePrompt");
    }
    const budget = maxTokens || 128;
    const userText = withLengthBudget(prompt, budget);
    const packed =
      system && system.trim()
        ? `${system.trim()}\n\n${userText}`
        : userText;

    // 8×8 mid-gray RGB — enough for the vision tower; not a real scene.
    const placeholder = new RawImage(
      new Uint8ClampedArray(8 * 8 * 3).fill(128),
      8,
      8,
      3
    );

    const conversation = [
      {
        role: "user",
        content: [
          { type: "image" },
          { type: "text", text: packed },
        ],
      },
    ];
    const text = processor.apply_chat_template(conversation, {
      add_generation_prompt: true,
    });
    const inputs = await processor(text, placeholder);
    return generateFromInputs({
      processor,
      model,
      inputs,
      maxTokens: budget,
      temperature,
      onToken,
    });
  }

  return {
    MAX_IMAGE_EDGE,
    VIDEO_MIN_PIXELS,
    VIDEO_MAX_PIXELS,
    DEFAULT_MODEL_ID,
    DEFAULT_DTYPE,
    resolveEosTokenIds,
    withLengthBudget,
    resizeLongEdge,
    videoSmartResize,
    loadVlModel,
    describeImage,
    describeVideo,
    buildVlVideoInputs,
    generatePrompt,
    generateFromInputs,
  };
}
