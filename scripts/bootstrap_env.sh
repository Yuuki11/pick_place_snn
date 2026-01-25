#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-visrl}"
MANISKILL_TAG="v3.0.0b22"
MANISKILL_COMMIT="33967b9e3ead1f841eec57cc9f31d0d8b8cf0907"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but not found on PATH." >&2
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -n "${ENV_NAME}" python=3.11 -y
fi

conda run -n "${ENV_NAME}" python -m pip install --upgrade pip setuptools wheel
conda run -n "${ENV_NAME}" python -m pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
conda run -n "${ENV_NAME}" python -m pip install "mani-skill==3.0.0b22" tensorboard pandas imageio imageio-ffmpeg pyyaml rich pytest

if [[ ! -d "${ROOT_DIR}/third_party/ManiSkill" ]]; then
  git clone --depth 1 --branch "${MANISKILL_TAG}" https://github.com/haosulab/ManiSkill.git "${ROOT_DIR}/third_party/ManiSkill"
fi

ACTUAL_COMMIT="$(git -C "${ROOT_DIR}/third_party/ManiSkill" rev-parse HEAD)"
if [[ "${ACTUAL_COMMIT}" != "${MANISKILL_COMMIT}" ]]; then
  echo "Pinned ManiSkill source mismatch: expected ${MANISKILL_COMMIT}, got ${ACTUAL_COMMIT}" >&2
  exit 1
fi

conda run -n "${ENV_NAME}" python -m pip install -e "${ROOT_DIR}[dev]"

echo "Bootstrap complete."
echo "Next:"
echo "  conda activate ${ENV_NAME}"
echo "  python ${ROOT_DIR}/scripts/validate_headless.py"
