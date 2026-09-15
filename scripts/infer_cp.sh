#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
"${PYTHON:-python}" infer_cp.py \
  --config configs/composer_performer.json \
  --checkpoint checkpoints/composer_performer.safetensors \
  --prompt-audio examples/piano/prompt.wav \
  --prompt-midi examples/piano/prompt.mid \
  --target-midi examples/piano/target.mid \
  --output outputs/example_cp.wav \
  --steps 4 --cfg-scale 1 --schedule cosine \
  --seed 114 --max-release-duration 3 "$@"
