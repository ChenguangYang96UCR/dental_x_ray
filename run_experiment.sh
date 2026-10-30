#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${1:?Usage: ./run_experiment.sh DATA_DIR [OUTPUT_DIR]}"
OUTPUT_DIR="${2:-${SCRIPT_DIR}/outputs}"
MODELS_TEXT="${MODELS:-resnet18 dinov3_vits16 siglip2}"
CLASSES_CSV="${CLASSES:-Healthy Teeth,Caries}"
FOLDS="${FOLDS:-5}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DEVICE="${DEVICE:-auto}"

if ! command -v python >/dev/null 2>&1; then
  echo "python was not found. Activate the intended Conda or virtual environment first." >&2
  exit 1
fi

read -r -a MODEL_ARGS <<< "${MODELS_TEXT}"
IFS=',' read -r -a CLASS_ARGS <<< "${CLASSES_CSV}"

echo "Python: $(command -v python)"
python -c \
  "import torch; print(f'Torch: {torch.__version__}, CUDA: {torch.version.cuda}, available: {torch.cuda.is_available()}')"

python "${SCRIPT_DIR}/run_feature_benchmark.py" \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --models "${MODEL_ARGS[@]}" \
  --classes "${CLASS_ARGS[@]}" \
  --folds "${FOLDS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  --seed 42 \
  --image-size 224 \
  --pca-dim 64

