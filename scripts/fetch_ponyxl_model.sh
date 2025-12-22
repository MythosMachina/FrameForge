#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ID="${1:-darkstorm2150/pony-diffusion-xl-base-1.0}"
FILENAME="${2:-ponyDiffusionV6XL.safetensors}"
DEST_DIR="${ROOT_DIR}/trainer/models"

if [[ -x "${ROOT_DIR}/.venv/bin/python3" ]]; then
  PYTHON="${ROOT_DIR}/.venv/bin/python3"
else
  PYTHON="python3"
fi

mkdir -p "${DEST_DIR}"

REPO_ID="${REPO_ID}" FILENAME="${FILENAME}" DEST_DIR="${DEST_DIR}" "${PYTHON}" - <<'PY'
import os
from huggingface_hub import hf_hub_download

repo_id = os.environ["REPO_ID"]
filename = os.environ["FILENAME"]
dest_dir = os.environ["DEST_DIR"]
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    token=token,
    local_dir=dest_dir,
    local_dir_use_symlinks=False,
)
print(path)
PY

echo "PonyXL model downloaded to: ${DEST_DIR}/${FILENAME}"
