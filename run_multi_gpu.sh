#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_DATA_DIR="${SCRIPT_DIR}/data/Dental OPG XRAY Dataset/Dental OPG (Classification)"
DATA_DIR="${1:-${DEFAULT_DATA_DIR}}"
OUTPUT_ROOT="${2:-${SCRIPT_DIR}/outputs_binary_multi_gpu}"
MODELS_TEXT="${MODELS:-resnet18 dinov3_vits16 siglip2}"
CLASSES_CSV="${CLASSES:-Healthy Teeth,Caries}"
GPUS_CSV="${GPUS:-}"
FOLDS="${FOLDS:-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"

if ! command -v python >/dev/null 2>&1; then
  echo "python was not found. Activate the intended Conda or virtual environment first." >&2
  exit 1
fi

if [[ -z "${GPUS_CSV}" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi was not found. Set GPUS explicitly, for example GPUS=0,1." >&2
    exit 1
  fi
  GPUS_CSV="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
fi

IFS=',' read -r -a GPU_ARGS <<< "${GPUS_CSV}"
read -r -a MODEL_ARGS <<< "${MODELS_TEXT}"
IFS=',' read -r -a CLASS_ARGS <<< "${CLASSES_CSV}"

if [[ "${#GPU_ARGS[@]}" -eq 0 ]]; then
  echo "No GPUs were selected." >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}/workers" "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/merged"

echo "GPUs: ${GPU_ARGS[*]}"
echo "Models: ${MODEL_ARGS[*]}"
echo "Data: ${DATA_DIR}"
echo "Python: $(command -v python)"
python -c \
  "import torch; print(f'Torch: {torch.__version__}, CUDA: {torch.version.cuda}, available: {torch.cuda.is_available()}')"

# Run in waves. Each wave assigns at most one model to each physical GPU, so a
# GPU never hosts two large encoders simultaneously.
model_count="${#MODEL_ARGS[@]}"
gpu_count="${#GPU_ARGS[@]}"

for ((start=0; start<model_count; start+=gpu_count)); do
  pids=()
  names=()

  for ((slot=0; slot<gpu_count; slot++)); do
    model_index=$((start + slot))
    if (( model_index >= model_count )); then
      break
    fi

    model_name="${MODEL_ARGS[model_index]}"
    gpu_id="${GPU_ARGS[slot]}"
    worker_output="${OUTPUT_ROOT}/workers/${model_name}"
    log_file="${OUTPUT_ROOT}/logs/${model_name}.log"

    echo "Starting ${model_name} on physical GPU ${gpu_id}"
    (
      CUDA_VISIBLE_DEVICES="${gpu_id}" \
      python "${SCRIPT_DIR}/run_feature_benchmark.py" \
        --data-dir "${DATA_DIR}" \
        --output-dir "${worker_output}" \
        --models "${model_name}" \
        --classes "${CLASS_ARGS[@]}" \
        --folds "${FOLDS}" \
        --batch-size "${BATCH_SIZE}" \
        --num-workers "${NUM_WORKERS}" \
        --device cuda \
        --seed 42 \
        --image-size 224 \
        --pca-dim 64
    ) >"${log_file}" 2>&1 &

    pids+=("$!")
    names+=("${model_name}")
  done

  wave_failed=0
  for index in "${!pids[@]}"; do
    if wait "${pids[index]}"; then
      echo "Finished ${names[index]}"
    else
      echo "Failed ${names[index]}; see ${OUTPUT_ROOT}/logs/${names[index]}.log" >&2
      wave_failed=1
    fi
  done
  if (( wave_failed != 0 )); then
    exit 2
  fi
done

python "${SCRIPT_DIR}/merge_benchmark_results.py" \
  --input-root "${OUTPUT_ROOT}/workers" \
  --output-dir "${OUTPUT_ROOT}/merged"

echo "Logs: ${OUTPUT_ROOT}/logs"
echo "Merged summary: ${OUTPUT_ROOT}/merged/summary.csv"

