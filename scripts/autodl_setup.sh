#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python - <<'PY'
import sys
print(f"python={sys.version.split()[0]}")
PY

python -m pip install --upgrade pip
python -m pip install -r requirements-autodl.txt

python - <<'PY'
import torch
import torchvision

print(f"torch={torch.__version__}")
print(f"torchvision={torchvision.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_runtime={torch.version.cuda}")
    print(f"gpu={torch.cuda.get_device_name(0)}")
PY

mkdir -p data checkpoints outputs/results outputs/figures outputs/logs
