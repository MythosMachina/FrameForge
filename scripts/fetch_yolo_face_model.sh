#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_URL="${1:-https://github.com/YapaLab/yolo-face/releases/download/v0.0.0/yolov12n-face.pt}"
DEST_DIR="${ROOT_DIR}/_system/models/face"
DEST_PATH="${DEST_DIR}/yolov12n-face.pt"

mkdir -p "${DEST_DIR}"

if command -v curl >/dev/null 2>&1; then
  curl -fL "${MODEL_URL}" -o "${DEST_PATH}"
elif command -v wget >/dev/null 2>&1; then
  wget -O "${DEST_PATH}" "${MODEL_URL}"
else
  echo "Error: curl or wget required to download model." >&2
  exit 1
fi

echo "YOLO face model downloaded to: ${DEST_PATH}"
