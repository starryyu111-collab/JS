#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

mkdir -p outputs/logs
python -m src.experiments.run_all_experiments --mode run-all 2>&1 | tee outputs/logs/autodl_run_all.log
bash scripts/autodl_package_results.sh
