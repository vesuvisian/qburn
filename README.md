<p align="center">
  <img src="web/static/logo.png" alt="qburn" width="112" />
</p>

# qburn

Run [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) with Burn (`0.23.0-pre.2`): hand-written vision and language modules, a native CLI, and a WebGPU browser demo that shares the same crate.

You pass an image and a short prompt; the model streams a caption. Nothing here wraps ONNX Runtime or transformers.js — the stack is Burn end to end. Architecture notes: [how it works](web/static/how-it-works.html). Why not the ONNX path: [engineering](web/static/engineering.html).

Weights are not in git. Convert Hugging Face safetensors to burnpack shards offline, then host them on the Hub (or any CORS-enabled origin).

## Layout

```text
crates/qburn/          Model, preprocess, generate, CLI
crates/converter/      HF safetensors → quantized .bpk shards + manifest.json
web/                   wasm-bindgen crate (WebGPU)
web/static/            Demo UI + engineering / how-it-works pages
web/pkg/               wasm-pack output (gitignored; built in CI / locally)
patches/               cubecl-runtime patch for wasm Burn 0.23
.github/workflows/     GitHub Pages deploy
scripts/               HF logits dump for parity checks
tests/parity/          Optional reference dumps
weights/               Local HF checkout + burnpacks (gitignored)
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

## Setup: download + convert weights

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

Loads the model once, then captions each image. Tokens always stream to stdout.

| Flag / arg | Default | Description |
|------------|---------|-------------|
| `--model <DIR>` | required | HF or burnpack directory. |
| `<PATHS>…` | required | Image files and/or directories. |
| `-r`, `--recursive` | off | Recurse when expanding folders. |
| `--prompt <TEXT>` | `Describe this scene.` | Caption instruction. |
| `--max-tokens <N>` | `128` | Generation budget. |
| `--temperature <F>` | `0` | Softmax temperature; `0` = greedy. |
| `--metrics` | off | Phase timings on stderr (below). |
| `--probe` | off | Dump prefill first-token top-10 logits to stderr. |

Directories contribute PNG/JPEG/WebP files (non-recursive unless `--recursive`). Explicit file paths are always queued.

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

# Same quantization as the browser demo
cargo run -p qburn --release -- describe \
  --model weights/burnpack-q4-mix \
  --prompt "Describe this scene." \
  path/to/photo.jpg
```

### `config`

Prints the built-in 2B config as JSON.

```bash
cargo run -p qburn --release -- config
```

## Browser demo

The page under `web/static/` loads wasm from `web/` (Burn `webgpu` via wasm-bindgen). A Web Worker owns the model so the UI stays responsive; tokens stream back with `postMessage`. Extra pages: [engineering](web/static/engineering.html), [how it works](web/static/how-it-works.html).

WebGPU is required — without `navigator.gpu` the page errors out (no CPU fallback for this model). Weights come from a burnpack `manifest.json` (Hub by default, or local — see below). Shards are fetched in order, concatenated, and cached for reloads.

Prefer `q4-mix` (~1.8 GiB) for quality; uniform `q4` (~1.3 GiB) is smaller. Fp16/fp32 packs are too large for typical wasm32 heaps once expanded. Temperature defaults to `0` (greedy) in the UI.

Expect the browser path to be much slower than the CLI. Shared memory on WebGPU is often ~32 KiB, and Burn’s hardware-aware CubeCL path has to live with that (see [patches/](patches/README.md)). A short caption can take a minute or more. Serve over `http://localhost` — `file://` will not load wasm.

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

After changing only `web/static/*`, re-run `bash web/stage.sh`. After changing `web/src`, `crates/qburn`, or `patches/`, re-run `wasm-pack` then `stage.sh`. Convert HF weights before using `--weights` (see [Setup](#setup-download--convert-weights)).

Override the manifest without the preview flag via `?manifest=…` or:

```html
<script>
  window.QBURN_MANIFEST =
    "https://huggingface.co/<you>/qburn-weights/resolve/main/manifest.json";
</script>
```

### GitHub Pages

Set the repo’s Pages source to **GitHub Actions**. [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml) builds the WebGPU wasm pack, stages `web/dist`, rejects any artifact file ≥100 MB, and deploys. Model weights stay on the Hub — only the SPA + wasm ship with Pages.

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
