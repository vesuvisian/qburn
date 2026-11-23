#!/usr/bin/env bash
# Stage static assets + wasm pkg into web/dist for Pages / local preview.
#
# Usage (from repo root, after wasm-pack):
#   bash web/stage.sh
#   bash web/stage.sh /path/to/alternate/pkg
#
# Output: web/dist/{index.html,engineering.html,how-it-works.html,demo.js,demo.css,worker.js,pkg/...}
# Preview: python3 web/preview.py
# Local weights: python3 web/preview.py --weights weights/burnpack-q4
#   (serves burnpacks at /weights/; does not copy them into dist)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIST="$ROOT/web/dist"
PKG_SRC="${1:-$ROOT/web/pkg}"

rm -rf "$DIST"
mkdir -p "$DIST/pkg"
cp -R "$ROOT/web/static/." "$DIST/"
if [[ -d "$PKG_SRC" ]]; then
  cp -R "$PKG_SRC/." "$DIST/pkg/"
else
  echo "warn: $PKG_SRC missing — run wasm-pack first" >&2
fi

# Never stage Hub weight shards into Pages artifacts.
if find "$DIST" -type f -size +99M 2>/dev/null | grep -q .; then
  echo "error: file ≥100MB under $DIST — weights belong on Hugging Face Hub, not Pages" >&2
  find "$DIST" -type f -size +99M -print >&2
  exit 1
fi

echo "staged -> $DIST"
