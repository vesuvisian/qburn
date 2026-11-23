#!/usr/bin/env python3
"""Local static server with correct application/wasm MIME and optional weights.

GitHub Pages serves `.wasm` as `application/wasm`. Python's default
`SimpleHTTPRequestHandler` often does not, which breaks
`WebAssembly.instantiateStreaming` (slow fallback or hard failure).

Uses ``ThreadingTCPServer`` so multi-hundred-MB shard downloads do not block
concurrent requests (SPA assets, HEAD probes, Cache API).

Usage (after `bash web/stage.sh`):

    python3 web/preview.py                              # Hub weights (default)
    python3 web/preview.py --weights weights/burnpack-q4-mix
    python3 web/preview.py --port 8080
    python3 web/preview.py --dir web/dist

With `--weights DIR`, burnpack files are served at `/weights/` (same origin).
The demo auto-detects `./weights/manifest.json` when present.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import posixpath
import socketserver
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "dist"
REPO = Path(__file__).resolve().parent.parent


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **getattr(http.server.SimpleHTTPRequestHandler, "extensions_map", {}),
        ".wasm": "application/wasm",
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".json": "application/json",
        ".css": "text/css",
        ".html": "text/html",
        ".part": "application/octet-stream",
        ".bpk": "application/octet-stream",
    }

    # Set by partial() below.
    weights_dir: Path | None = None

    def translate_path(self, path: str) -> str:
        parsed = urllib.parse.urlsplit(path)
        rel = posixpath.normpath(urllib.parse.unquote(parsed.path))
        if rel.startswith("/weights/") or rel == "/weights":
            if self.weights_dir is None:
                return super().translate_path(path)
            suffix = rel[len("/weights") :].lstrip("/")
            # Block path traversal outside the weights directory.
            candidate = (self.weights_dir / suffix).resolve()
            try:
                candidate.relative_to(self.weights_dir.resolve())
            except ValueError:
                return str(self.weights_dir / "__denied__")
            return str(candidate)
        return super().translate_path(path)


def default_weights() -> Path | None:
    """Prefer q4-mix, then q4, then generic burnpack dirs under repo `weights/`."""
    for name in ("burnpack-q4-mix", "burnpack-q4", "burnpack"):
        cand = REPO / "weights" / name / "manifest.json"
        if cand.is_file():
            return cand.parent
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--dir",
        type=Path,
        default=ROOT,
        help="Directory to serve (default: web/dist)",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        nargs="?",
        const=Path("__auto__"),
        default=None,
        help=(
            "Burnpack directory (manifest.json + shards) served at /weights/. "
            "Pass --weights alone to auto-pick weights/burnpack-q4-mix, "
            "burnpack-q4, or burnpack."
        ),
    )
    args = parser.parse_args()
    directory = args.dir.resolve()
    if not directory.exists():
        raise SystemExit(
            f"{directory} missing — run: bash web/stage.sh after wasm-pack build"
        )

    weights_dir: Path | None = None
    if args.weights is not None:
        if str(args.weights) == "__auto__":
            weights_dir = default_weights()
            if weights_dir is None:
                raise SystemExit(
                    "no local burnpack found — convert first, e.g.\n"
                    "  cargo run -p qburn-converter --release -- \\\n"
                    "    --model-dir weights/Qwen3-VL-2B-Instruct \\\n"
                    "    --out-dir weights/burnpack-q4-mix --quant q4-mix"
                )
        else:
            weights_dir = args.weights.resolve()
        manifest = weights_dir / "manifest.json"
        if not manifest.is_file():
            raise SystemExit(f"missing {manifest} — point --weights at a converter out-dir")

    handler = functools.partial(Handler, directory=str(directory))
    Handler.weights_dir = weights_dir

    # Threaded: large shard GETs must not block HEAD/manifest/wasm while loading.
    with socketserver.ThreadingTCPServer(("", args.port), handler) as httpd:
        httpd.daemon_threads = True
        print(f"serving {directory} at http://127.0.0.1:{args.port}/")
        if weights_dir is not None:
            print(f"local weights: {weights_dir} → http://127.0.0.1:{args.port}/weights/")
            print(
                f"manifest: http://127.0.0.1:{args.port}/weights/manifest.json "
                "(demo auto-detects this)"
            )
        else:
            print("weights: Hub default (pass --weights [DIR] for local burnpacks)")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
