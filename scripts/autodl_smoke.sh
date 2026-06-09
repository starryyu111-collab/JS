#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

python -m pytest -q
python -m src.experiments.train_fp32 \
  --epochs 1 \
  --batch-size 64 \
  --num-workers 2 \
  --max-train-batches 1 \
  --max-val-batches 1 \
  --max-test-batches 1
