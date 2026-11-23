# Parity fixtures

Numeric checks against Hugging Face Transformers for Milestone 1 (text) and Milestone 2 (vision).

## Layout

```text
tests/parity/
  prompt_ids.json       # optional: token ids for a fixed chat prompt
  logits_ref.json       # from scripts/dump_logits_ref.py (topk ids + logits)
  describe_ids.json     # optional: multimodal token ids for a fixture image
```

These files are **not** required for `cargo test` CI smoke tests (`config_parity`, `preprocess`). Add them locally when debugging weight load or RoPE.

## Dump HF references

```bash
# Needs: pip install torch transformers (in a venv)
python scripts/dump_logits_ref.py \
  --model weights/Qwen3-VL-2B-Instruct \
  --prompt "Hello" \
  --out tests/parity/logits_ref.json
```

## Native checks

```bash
# Always-on (no weights)
cargo test -p qburn --test config_parity --test preprocess

# With a local HF checkout
cargo run -p qburn --release -- prompt \
  --model weights/Qwen3-VL-2B-Instruct \
  --stream "Hello"

cargo run -p qburn --release -- describe \
  --model weights/Qwen3-VL-2B-Instruct \
  path/to/fixture.jpg
```

Compare greedy text / top-k token ids to `logits_ref.json`. Expect ~1e-4 agreement on fp32 logits when implementations match; fp16 / Q4 packs will diverge more — judge by caption quality for those.

### Prefill probe (Metal vs WebGPU)

```bash
cargo run -p qburn --release --no-default-features --features metal,cli -- \
  describe --model weights/burnpack-q4-mix --temperature 0 --max-tokens 64 --probe \
  path/to/photo.jpg

# No-fusion path (falls back to Auto wgpu if host lacks WebGPU graphics API)
cargo run -p qburn --release --no-default-features --features webgpu,cli -- \
  describe --model weights/burnpack-q4-mix --temperature 0 --max-tokens 64 --probe \
  path/to/photo.jpg
```

Compare stderr `image grid … merge tokens` and `prefill probe:` top-10. The browser worker logs the same probe after each describe.

## Burnpack round-trip

```bash
cargo run -p qburn-converter --release -- \
  --model-dir weights/Qwen3-VL-2B-Instruct \
  --out-dir /tmp/qburn-fp16 --quant fp16 --shard-mib 512 --keep-full

# Load via library (see qburn::load::load_from_burnpack) and re-run the same prompt.
```
