/**
 * Shared transformers.js helpers for the browser worker and Node CLI.
 *
 * Import transformers bindings in the host, then call createTjsInfer(api).
 * That keeps CDN vs npm resolution out of this file.
 */

/** Long-edge resize so vision memory stays in a browser-friendly band. */
export const MAX_IMAGE_EDGE = 448;

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
    else return prompt;

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
    const streamer = new TextStreamer(processor.tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      // Do not gate on sawEos: token_callback may mark EOS in the same step
      // as the final text piece, and dropping it leaves short replies blank.
      callback_function: (piece) => {
        if (piece && onToken) onToken(piece);
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
    return { text: full, newTokenCount };
  }

  /**
   * @param {object} opts
   * @param {*} opts.processor
   * @param {*} opts.model
   * @param {*} opts.image — RawImage
   * @param {string} [opts.prompt]
   * @param {number} [opts.maxTokens]
   * @param {number} [opts.temperature]
   * @param {(piece: string) => void} [opts.onToken]
   */
  async function describeImage({
    processor,
    model,
    image,
    prompt,
    maxTokens,
    temperature,
    onToken,
  }) {
    const budget = maxTokens || 128;
    const userText = withLengthBudget(prompt || "Describe this scene.", budget);

    const conversation = [
      {
        role: "user",
        content: [
          { type: "image" },
          { type: "text", text: userText },
        ],
      },
    ];
    const text = processor.apply_chat_template(conversation, {
      add_generation_prompt: true,
    });
    const inputs = await processor(text, image);
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
    DEFAULT_MODEL_ID,
    DEFAULT_DTYPE,
    resolveEosTokenIds,
    withLengthBudget,
    resizeLongEdge,
    loadVlModel,
    describeImage,
    generatePrompt,
    generateFromInputs,
  };
}
