#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

log() {
  echo "[models] $*"
}

log "downloading tagger model"
"${ROOT_DIR}/scripts/fetch_tagger_model.sh"

log "downloading face detection model"
"${ROOT_DIR}/scripts/fetch_yolo_face_model.sh"

log "downloading PonyXL training model"
"${ROOT_DIR}/scripts/fetch_ponyxl_model.sh"

log "model downloads complete"
