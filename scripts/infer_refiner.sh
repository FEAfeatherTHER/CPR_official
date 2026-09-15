#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
"${PYTHON:-python}" infer_refiner.py \
  --config configs/refiner.json \
  --input outputs/example_cp.wav \
  --output outputs/example_cpr.wav "$@"
