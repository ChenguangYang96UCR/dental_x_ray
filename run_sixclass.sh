#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_DATA_DIR="${SCRIPT_DIR}/data/Dental OPG XRAY Dataset/Dental OPG (Classification)"
DATA_DIR="${1:-${DEFAULT_DATA_DIR}}"
OUTPUT_DIR="${2:-${SCRIPT_DIR}/outputs_sixclass}"

export CLASSES="BDC-BDR,Caries,Fractured Teeth,Healthy Teeth,Impacted Teeth,Infection"
export MODELS="${MODELS:-resnet18 dinov3_vits16 siglip2}"
export FOLDS="${FOLDS:-5}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export DEVICE="${DEVICE:-auto}"

exec "${SCRIPT_DIR}/run_experiment.sh" "${DATA_DIR}" "${OUTPUT_DIR}"

