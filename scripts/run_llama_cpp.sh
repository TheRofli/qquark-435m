#!/usr/bin/env bash
set -e

MODEL="${1:-release/qquark-435m-v0.1-byte-Q4_K_M.gguf}"
PORT="${PORT:-8088}"
CTX="${CTX:-2048}"

if [ ! -f "$MODEL" ]; then
  echo "Model not found: $MODEL"
  echo "Download qquark-435m-v0.1-byte-Q4_K_M.gguf from the release page and place it in release/"
  exit 1
fi

llama-server \
  -m "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --ctx-size "$CTX" \
  --n-gpu-layers 999 \
  --flash-attn on
