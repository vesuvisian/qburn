<p align="center">
  <img src="web/static/logo.png" alt="qburn" width="112" />
</p>

# qburn

[Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) on Burn (`0.23.0-pre.2`) — hand-written vision and language modules, a native CLI, and a WebGPU browser demo that share the same crate. No ONNX in the middle.

Drop in a photo, ask for a description, and the model streams a caption. Try it live at [vesuvisian.github.io/qburn](https://vesuvisian.github.io/qburn/), or read the [architecture](web/static/architecture.html) and [Burn vs transformers.js](web/static/compare.html) notes.

Weights are not in git. Convert Hugging Face safetensors to burnpack shards offline, then host them on the Hub (or any CORS-enabled origin).

## Contents

- [Layout](#layout)
- [Setup](#setup)
- [GPU backends](#gpu-backends)
- [CLI](#cli)
- [Browser demo](#browser-demo)
- [Comparison (transformers.js)](#comparison-transformersjs)
- [Parity / testing](#parity--testing)
- [Implementation notes](#implementation-notes)

## Layout

```text
crates/qburn/          Model, preprocess, generate, CLI
crates/converter/      HF safetensors → quantized .bpk shards + manifest.json
web/                   wasm-bindgen crate (WebGPU)
web/static/            Intro, demos, architecture, compare
                       index · demo · tjs · architecture · compare
web/pkg/               wasm-pack output (gitignored; built in CI / locally)
patches/               cubecl-runtime patch for wasm Burn 0.23
.github/workflows/     GitHub Pages deploy
scripts/               HF logits dump; transformers.js Node CLI (`tjs-cli/`)
tests/parity/          Optional reference dumps
weights/               Local HF checkout + burnpacks (gitignored)
```

## Setup

1. Get a local checkout of `Qwen/Qwen3-VL-2B-Instruct` (`config.json`, `tokenizer.json`, `model.safetensors`, …), or let the converter download it.
2. Convert to burnpack shards with [`qburn-converter`](crates/converter):

```bash
# q4-mix is the usual choice for browser quality (F16 where it matters, Q4S on bulky linears)
cargo run -p qburn-converter --release -- \
  --hf-repo Qwen/Qwen3-VL-2B-Instruct \
  --model-dir weights/Qwen3-VL-2B-Instruct \
  --out-dir weights/burnpack-q4-mix \
  --quant q4-mix \
  --shard-mib 512
```

### Converter flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model-dir <DIR>` | required | Local HF checkout (or download target). |
| `--out-dir <DIR>` | required | Writes `manifest.json`, tokenizer, config, and shards. |
| `--hf-repo <ID>` | — | Optional Hub id to download into `--model-dir` first. |
| `--quant <MODE>` | `fp16` | `fp32` (~9 GiB), `fp16` (~4.5 GiB), `q4` (~1.3 GiB), or `q4-mix` (~1.8 GiB). |
| `--shard-mib <N>` | `512` | Max shard size in MiB. |
| `--keep-full` | off | Keep the unsharded `.bpk` after writing `.bpk.part` files. |

`q4-mix` quantizes the large decoder and vision-block linears to Q4S (block 32, Burn MinMax — not AWQ/GPTQ) and leaves embeddings, norms, and vision patch/pos/merger/DeepStack in F16. Uniform `q4` is smaller and lower quality.

Shards look like `model-*-NNNNN-of-MMMMM.bpk.part` and are listed in order in `manifest.json`. Upload the whole out-dir to a Hub repo with CORS. Do not commit packs to git (GitHub’s 100 MB limit).

```text
weights/burnpack-q4-mix/
├── config.json
├── tokenizer.json
├── manifest.json
├── model-q4-mix-00000-of-00004.bpk.part
├── …
└── model-q4-mix-00003-of-00004.bpk.part
```

## GPU backends

The library, CLI, and wasm demo share feature flags. The CLI default is `wgpu` + `cli` + `autotune` (GPU features also pull in `burn/fusion`).

| Feature | Backend | Notes |
|---------|---------|--------|
| `wgpu` (default) | Burn wgpu | Cross-platform GPU. On macOS this usually rides Metal *through* wgpu; Burn 0.23 may see ~32 KiB shared memory and pick slower matmuls than native `metal`. |
| `metal` | Burn CubeCL Metal | Native Metal (`Device::metal`) + fusion + autotune. Prefer on Apple Silicon. |
| `wgpu-msl` | Burn wgpu → MSL | wgpu compiling to Metal shading language; usually still slower than `metal`. |
| `cuda` | Burn CubeCL CUDA | Native CUDA (`Device::cuda(0)`) + fusion + autotune. |
| `flex` | Burn flex | CPU only — fine for tiny smoke tests, not for this VLM. |
| `webgpu` | Burn webgpu / WGSL | Browser WebGPU. No fusion, no autotune. Used by `web/`. |
| `autotune` | — | Kernel autotune. On by default for native; omit for wasm. |
| `cli` | — | Builds the `qburn` binary (`clap`). |

Always use `--release` for real inference. After a Burn upgrade the first run may re-autotune; later runs reuse the cache.

```bash
# Default wgpu CLI
cargo run -p qburn --release -- prompt --model weights/Qwen3-VL-2B-Instruct "Hello"

# Native Metal (macOS) — usually faster than default wgpu on Apple Silicon
cargo run -p qburn --release --no-default-features --features metal,cli -- \
  describe --model weights/burnpack-q4-mix photo.jpg
```

## CLI

```bash
cargo run -p qburn --release -- <command> …
```

`--model` accepts either an HF safetensors directory or a converter out-dir (`manifest.json` plus `.bpk.part` / `.bpk`). Point at a burnpack when you want the same quantization as the browser demo.

### `prompt` — text only

| Flag / arg | Default | Description |
|------------|---------|-------------|
| `--model <DIR>` | required | HF or burnpack directory. |
| `<PROMPT>` | required | User prompt (positional). |
| `--system <TEXT>` | `You are a helpful assistant.` | System message. |
| `--max-tokens <N>` | `128` | Generation budget. |
| `--temperature <F>` | `0` | Softmax temperature; `0` = greedy. |
| `--stream` | off | Print tokens as they arrive. |
| `--metrics` | off | Print TTFT / total / decode tok/s to stderr. |

```bash
cargo run -p qburn --release -- prompt \
  --model weights/Qwen3-VL-2B-Instruct \
  --temperature 0 \
  --stream "Say hello in one sentence."
```

### `describe` — scene description

Loads the model once, then captions each image (or all images in one prompt with `--together`), or describes a **single video** (Qwen3-VL timestamped `<|video_pad|>` blocks). Tokens always stream to stdout.

| Flag / arg | Default | Description |
|------------|---------|-------------|
| `--model <DIR>` | required | HF or burnpack directory. |
| `<PATHS>…` | required | Image files and/or directories, or one video. |
| `-r`, `--recursive` | off | Recurse when expanding folders (images only). |
| `--together` | off | One multi-image prompt over all paths (default: caption each file). Image-only. |
| `--video` | off | Treat the single path as one video (container or ordered frame directory). |
| `--fps <F>` | `2` | Sample this many frames per second (video only; exclusive with `--num-frames`). |
| `--num-frames <N>` | — | Sample exactly N frames (video only; exclusive with `--fps`). Max 16. |
| `--prompt <TEXT>` | `Describe this scene.` | Caption instruction. |
| `--max-tokens <N>` | `128` | Generation budget. |
| `--temperature <F>` | `0` | Softmax temperature; `0` = greedy. |
| `--metrics` | off | Phase timings on stderr (below). |
| `--probe` | off | Dump prefill first-token top-10 logits to stderr. |

Directories contribute PNG/JPEG/WebP files (non-recursive unless `--recursive`). Explicit file paths are always queued. Video containers (`mp4`, `webm`, `mov`, `mkv`, `avi`) require `ffmpeg` on `PATH` (and prefer `ffprobe`). Frame folders use `--video` and skip ffmpeg. Caps: min 4 frames, max 16. Spatial resolution uses the HF video pixel budget (`t×h×w`), so longer clips shrink instead of stacking full-res frames. Do not mix images and video in one invoke.

With `--metrics`, stderr reports:

| Field | Meaning |
|-------|---------|
| `preprocess` | Decode / resize / patchify |
| `vision` | Vision encoder + multimodal splice |
| `prefill` | First decoder forward over the full sequence + first sample |
| `ttft` | `vision` + `prefill` (time until the first token) |
| `decode` | Steps after the first token |
| `tok/s` | Decode throughput after the first token |

Batch runs also print avg/min/max TTFT.

```bash
# Single image
cargo run -p qburn --release -- describe \
  --model weights/Qwen3-VL-2B-Instruct \
  path/to/photo.jpg

# Batch (files + folder), recurse, with latency metrics
cargo run -p qburn --release -- describe \
  --model weights/Qwen3-VL-2B-Instruct \
  --recursive \
  --metrics \
  a.jpg b.jpg photos/

# Multi-image in one prompt (order = path expansion order)
cargo run -p qburn --release -- describe \
  --model weights/Qwen3-VL-2B-Instruct \
  --together \
  --prompt "What changed between these photos?" \
  before.jpg after.jpg

# Same quantization as the browser demo
cargo run -p qburn --release -- describe \
  --model weights/burnpack-q4-mix \
  --prompt "Describe this scene." \
  path/to/photo.jpg

# Single video (ffmpeg on PATH)
cargo run -p qburn --release -- describe \
  --model weights/burnpack-q4-mix \
  --fps 2 \
  --prompt "What happens in this clip?" \
  clip.mp4

# Ordered frame directory as one video
cargo run -p qburn --release -- describe \
  --model weights/burnpack-q4-mix \
  --video \
  --num-frames 8 \
  frames/
```

### `config`

Prints the built-in 2B config as JSON.

```bash
cargo run -p qburn --release -- config
```

## Browser demo

Live: [vesuvisian.github.io/qburn](https://vesuvisian.github.io/qburn/) (intro at `/`, Burn demo at [`/demo.html`](https://vesuvisian.github.io/qburn/demo.html)). Narrative for the pipeline and quant choices lives in [architecture](web/static/architecture.html); keep this section for local build/preview.

The Burn page loads wasm from `web/` (Burn `webgpu` via wasm-bindgen). A Web Worker owns the model so the UI stays responsive; tokens stream back with `postMessage`. WebGPU is required — no CPU fallback. Weights come from a burnpack `manifest.json` (Hub by default, or local — see below). Prefer `q4-mix` (~1.8 GiB); uniform `q4` is smaller. Temperature defaults to `0` (greedy). Upload up to **4 images** for one multi-image prompt, **or one video** (**exactly 4** sampled frames, exclusive with images). The transformers.js page matches those limits.

**Video frame vs resolution tradeoff.** Qwen3-VL’s video `smart_resize` budgets pixels over `t×h×w`. More sampled frames shrink each frame spatially. The browser demos fix **4 frames** (`grid_t = 2` after `temporal_patch_size=2`) so captions stay spatially sharp under WebGPU latency; sampling 8 at 2 fps tends to crush detail into vague “abstract” captions. The CLI still exposes `--fps` / `--num-frames` (min 4, max 16) when you want more temporal coverage and accept smaller frames. Architecture notes: [architecture](web/static/architecture.html).

Expect the browser path to be much slower than the CLI (often a minute for a short caption). Shared memory on WebGPU is often ~32 KiB (see [patches/](patches/README.md)). Serve over `http://localhost` — `file://` will not load wasm.

Manifest resolution: `?manifest=` → `window.QBURN_MANIFEST` → `./weights/manifest.json` (local preview) → Hub default.

### Local preview

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
# Optional but recommended: Binaryen (`wasm-opt`) for a smaller/faster .wasm
#   macOS: brew install binaryen
#   Debian/Ubuntu: sudo apt-get install binaryen

# From the repo root — release wasm is slow the first time (~50 MB .wasm)
wasm-pack build web --release --target web --out-dir "$PWD/web/pkg" \
  -- --no-default-features --features webgpu

bash web/stage.sh            # copies static/ + pkg/ → web/dist/
python3 web/preview.py       # Hub weights by default
```

| Flag | Default | Description |
|------|---------|-------------|
| `--port <N>` | `8000` | Listen port. |
| `--dir <DIR>` | `web/dist` | Static root to serve. |
| `--weights [DIR]` | Hub | Serve burnpacks at `/weights/`. Omit `DIR` to auto-pick `weights/burnpack-q4-mix`, `burnpack-q4`, or `burnpack`. |

```bash
python3 web/preview.py --weights weights/burnpack-q4-mix
python3 web/preview.py --weights   # auto-pick local burnpack dir
```

`preview.py` uses a threaded HTTP server so large shard transfers do not stall the SPA, and sets `application/wasm` for `.wasm` files.

After changing only `web/static/*`, re-run `bash web/stage.sh`. After changing `web/src`, `crates/qburn`, or `patches/`, re-run `wasm-pack` then `stage.sh`. Convert HF weights before using `--weights` (see [Setup](#setup)).

Override the manifest without the preview flag via `?manifest=…` or:

```html
<script>
  window.QBURN_MANIFEST =
    "https://huggingface.co/<you>/qburn-weights/resolve/main/manifest.json";
</script>
```

### GitHub Pages

Deployed site: [vesuvisian.github.io/qburn](https://vesuvisian.github.io/qburn/). Set the repo’s Pages source to **GitHub Actions**. [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml) builds the WebGPU wasm pack, stages `web/dist`, rejects any artifact file ≥100 MB, and deploys. Model weights stay on the Hub — only the SPA + wasm ship with Pages.

## Comparison (transformers.js)

Side-by-side checks against Burn. Stack rationale: [compare](web/static/compare.html). Same ONNX model and shared helpers in [`web/static/tjs-infer.js`](web/static/tjs-infer.js) — not Burn.

### Browser page

[`web/static/tjs.html`](web/static/tjs.html) (live: [`/tjs.html`](https://vesuvisian.github.io/qburn/tjs.html)) runs the caption job through [@huggingface/transformers](https://huggingface.co/docs/transformers.js) (CDN) and ONNX Runtime WebGPU. It loads [`onnx-community/Qwen3-VL-2B-Instruct-ONNX`](https://huggingface.co/onnx-community/Qwen3-VL-2B-Instruct-ONNX) from the Hub (`?model=` overrides the id). No `wasm-pack` build is required; after editing `web/static/*`, re-stage and open `/tjs.html`.

### Node CLI

Local twin of the browser page (`prompt` + `describe`):

```bash
cd scripts/tjs-cli && npm install
# from repo root:
node scripts/tjs-cli/bin.mjs prompt --stream --max-tokens 64 "Say hello in one sentence."
node scripts/tjs-cli/bin.mjs describe --metrics path/to/photo.jpg
```

| Flag / command | Notes |
|----------------|--------|
| `prompt` / `describe` | Same verbs as the Burn CLI. |
| `--model <id\|path>` | Default Hub id `onnx-community/Qwen3-VL-2B-Instruct-ONNX`; local ONNX checkout also works. |
| `--device <name>` | Default `webgpu` (same as the browser page). Use `--device cpu` if WebGPU is unavailable. |
| `--max-tokens <N>` | Generation budget (default `128`). |
| `--temperature <F>` | Softmax temperature; `0` = greedy (default). |
| `--metrics` | Print TTFT / total / decode tok/s to stderr. |
| `describe --prompt` | Caption instruction (default `Describe this scene.`). |
| `describe -r` | Recurse when expanding directories. |
| `describe --together` | One multi-image prompt over all paths (default: caption each file). |
| `describe --video` | Single video: container file or ordered frame directory (`ffmpeg` for containers). |
| `describe --fps` / `--num-frames` | Video sampling (default 2 fps; max 16 frames). |
| `prompt --system` | System message (default `You are a helpful assistant.`). |
| `prompt --stream` | Stream tokens to stdout as they arrive. |

First run downloads ONNX weights into `weights/tjs-cache/` (gitignored; override with `QBURN_TJS_CACHE` or `TRANSFORMERS_CACHE`). Later runs reuse that cache instead of re-fetching from the Hub. Defaults match the browser page (`webgpu` + the shared `q4f16` dtype mix). Fall back with `--device cpu` on machines without WebGPU. `prompt` attaches a tiny placeholder image because this ONNX VL export expects vision inputs (Burn’s text-only path does not).

## Parity / testing

Config and preprocess unit tests run without weights:

```bash
cargo test -p qburn --test config_parity --test preprocess
```

For logits parity against Hugging Face Transformers, dump a reference with [`scripts/dump_logits_ref.py`](scripts/dump_logits_ref.py) into `tests/parity/` (see [`tests/parity/README.md`](tests/parity/README.md)).

## Implementation notes

Hugging Face stores vision `patch_embed` as a Conv3d `[out, in, t, h, w]`; the loader flattens it to a Burn Linear `[in·t·h·w, out]` (+ bias).

Browser WebGPU is CubeCL WGSL. Desktop `wgpu` on macOS is not the same path as `--features metal`, so treat numeric parity across backends carefully. On wasm, create the device with `Device::wgpu_options().init_async()` (`qburn::resolve_device_async`) before any tensors, and use `into_data_async` for GPU readbacks — sync `into_data` panics in the browser.

Tokenizers use `fancy-regex` instead of `onig` so the crate builds for `wasm32-unknown-unknown`. The workspace `[patch.crates-io]` pins [`patches/cubecl-runtime`](patches/README.md) so Burn 0.23 WebGPU builds on wasm; matmul uses upstream hardware-aware tile selection.
