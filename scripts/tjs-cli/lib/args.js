/**
 * Minimal argv parser for `qburn-tjs describe|prompt …`.
 */

function usage() {
  return `Usage:
  qburn-tjs describe [options] <paths>…
  qburn-tjs prompt   [options] <PROMPT>

Shared options:
  --model <id|path>     Hub id or local ONNX dir
                        (default: onnx-community/Qwen3-VL-2B-Instruct-ONNX)
  --max-tokens <N>      Generation budget (default: 128)
  --temperature <F>     Softmax temperature; 0 = greedy (default: 0)
  --device <name>       ORT device (default: webgpu; use cpu if WebGPU unavailable)
  --metrics             Print TTFT / total / tok/s to stderr

describe:
  -r, --recursive       Recurse when expanding directories
  --prompt <TEXT>       Caption instruction (default: Describe this scene.)

prompt:
  --system <TEXT>       System message (default: You are a helpful assistant.)
  --stream              Stream tokens to stdout as they arrive
`;
}

/**
 * @param {string[]} argv — process.argv.slice(2)
 */
export function parseArgs(argv) {
  if (argv.length === 0 || argv[0] === "-h" || argv[0] === "--help") {
    return { help: true };
  }

  const command = argv[0];
  if (command !== "describe" && command !== "prompt") {
    throw new Error(`unknown command: ${command}\n\n${usage()}`);
  }

  const opts = {
    command,
    model: null,
    maxTokens: 128,
    temperature: 0,
    device: "webgpu",
    metrics: false,
    recursive: false,
    prompt: "Describe this scene.",
    system: "You are a helpful assistant.",
    stream: false,
    paths: [],
    userPrompt: null,
  };

  const rest = argv.slice(1);
  const positionals = [];
  for (let i = 0; i < rest.length; i++) {
    const a = rest[i];
    if (a === "-h" || a === "--help") {
      return { help: true };
    }
    if (a === "--metrics") {
      opts.metrics = true;
      continue;
    }
    if (a === "-r" || a === "--recursive") {
      opts.recursive = true;
      continue;
    }
    if (a === "--stream") {
      opts.stream = true;
      continue;
    }
    if (a === "--model") {
      opts.model = needValue(rest, ++i, a);
      continue;
    }
    if (a === "--max-tokens") {
      opts.maxTokens = parseInt(needValue(rest, ++i, a), 10);
      if (!Number.isFinite(opts.maxTokens) || opts.maxTokens < 1) {
        throw new Error(`invalid --max-tokens`);
      }
      continue;
    }
    if (a === "--temperature") {
      opts.temperature = Number(needValue(rest, ++i, a));
      if (!Number.isFinite(opts.temperature)) {
        throw new Error(`invalid --temperature`);
      }
      continue;
    }
    if (a === "--device") {
      opts.device = needValue(rest, ++i, a);
      continue;
    }
    if (a === "--prompt") {
      opts.prompt = needValue(rest, ++i, a);
      continue;
    }
    if (a === "--system") {
      opts.system = needValue(rest, ++i, a);
      continue;
    }
    if (a.startsWith("-")) {
      throw new Error(`unknown option: ${a}\n\n${usage()}`);
    }
    positionals.push(a);
  }

  if (command === "describe") {
    if (positionals.length === 0) {
      throw new Error(`describe requires one or more image paths\n\n${usage()}`);
    }
    opts.paths = positionals;
  } else {
    if (positionals.length === 0) {
      throw new Error(`prompt requires a prompt string\n\n${usage()}`);
    }
    opts.userPrompt = positionals.join(" ");
  }

  return opts;
}

function needValue(argv, i, flag) {
  if (i >= argv.length) throw new Error(`missing value for ${flag}`);
  return argv[i];
}

export { usage };
