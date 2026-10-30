#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-${SCRIPT_DIR}/.venv}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
TORCH_VERSION="${TORCH_VERSION:-2.7.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.1}"
INSTALL_MAMBA="${INSTALL_MAMBA:-0}"

python -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel

"${VENV_DIR}/bin/python" -m pip install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  --index-url "${TORCH_INDEX_URL}"

"${VENV_DIR}/bin/python" -m pip install -r "${SCRIPT_DIR}/requirements.txt"

if [[ "${INSTALL_MAMBA}" == "1" ]]; then
  "${VENV_DIR}/bin/python" -m pip install packaging ninja
  MAMBA_KEEP_CUDA_BUILD=TRUE "${VENV_DIR}/bin/python" -m pip install \
    "mamba-ssm==2.3.1" \
    --no-build-isolation \
    --no-deps
  "${VENV_DIR}/bin/python" -c \
    "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('mamba-ssm OK')"
fi

echo "Environment ready: ${VENV_DIR}"
echo "Activate with: source ${VENV_DIR}/bin/activate"
if [[ "${INSTALL_MAMBA}" != "1" ]]; then
  echo "Mamba support is optional. Install it later with:"
  echo "  ${VENV_DIR}/bin/python -m pip install packaging ninja"
  echo "  MAMBA_KEEP_CUDA_BUILD=TRUE ${VENV_DIR}/bin/python -m pip install mamba-ssm==2.3.1 --no-build-isolation --no-deps"
fi

