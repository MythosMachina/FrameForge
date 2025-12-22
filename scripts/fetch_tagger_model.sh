#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ID="${1:-SmilingWolf/wd-eva02-large-tagger-v3}"
DEST_ROOT="${ROOT_DIR}/_system/webapp/tagger_models"
SAFE_NAME="$(echo "$REPO_ID" | tr '/\\' '-' | tr -cd 'A-Za-z0-9._-')"
DEST_DIR="${DEST_ROOT}/${SAFE_NAME}"

if [[ -x "${ROOT_DIR}/.venv/bin/python3" ]]; then
  PYTHON="${ROOT_DIR}/.venv/bin/python3"
else
  PYTHON="python3"
fi

mkdir -p "${DEST_DIR}"

REPO_ID="${REPO_ID}" DEST_DIR="${DEST_DIR}" "${PYTHON}" - <<'PY'
import os
from huggingface_hub import hf_hub_download

repo_id = os.environ["REPO_ID"]
dest_dir = os.environ["DEST_DIR"]
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

files = ["model.safetensors", "config.json", "selected_tags.csv"]
for name in files:
    path = hf_hub_download(
        repo_id=repo_id,
        filename=name,
        token=token,
        local_dir=dest_dir,
        local_dir_use_symlinks=False,
    )
    print(path)
PY

echo "Tagger model downloaded to: ${DEST_DIR}"
