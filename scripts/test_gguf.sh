#!/usr/bin/env bash
set -e

SERVER="${SERVER:-http://127.0.0.1:8088}"

python -m qquark.cli \
  --server "$SERVER" \
  --no-context \
  "сделай кнопку красивее"
