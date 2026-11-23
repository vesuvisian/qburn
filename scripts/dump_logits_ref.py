#!/usr/bin/env python3
"""Dump reference logits for Milestone 1 parity tests.

Requires a venv with `torch` and `transformers` that can load Qwen3-VL.

Usage:

  python scripts/dump_logits_ref.py \\
      --model weights/Qwen3-VL-2B-Instruct \\
      --prompt "Hello" \\
      --out tests/parity/logits_ref.json

See tests/parity/README.md and web/static/architecture.html#parity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--prompt", default="Hello")
    p.add_argument("--system", default="You are a helpful assistant.")
    p.add_argument("--out", type=Path, default=Path("tests/parity/logits_ref.json"))
    p.add_argument("--topk", type=int, default=32)
    args = p.parse_args()

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float32, device_map="cpu"
    )
    processor = AutoProcessor.from_pretrained(args.model)
    text = (
        f"<|im_start|>system\n{args.system}<|im_end|>\n"
        f"<|im_start|>user\n{args.prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = processor.tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, use_cache=False)
        logits = out.logits[0, -1].float()
    topk = torch.topk(logits, args.topk)
    payload = {
        "prompt_ids": inputs["input_ids"][0].tolist(),
        "topk_ids": topk.indices.tolist(),
        "topk_logits": topk.values.tolist(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
