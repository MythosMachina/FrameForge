#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEBAPP_DIR="${ROOT_DIR}/webapp"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
RUN_USER="${FF_USER:-$(whoami)}"
SYSTEMD_DIR="${ROOT_DIR}/systemd"
LOG_DIR="${ROOT_DIR}/_system/logs"
WEBAPP_LOG_DIR="${WEBAPP_DIR}/logs"
DB_PATH="${ROOT_DIR}/_system/db/db.sqlite"
DB_BROKER_URL="http://127.0.0.1:8799"
PATH_ENV="${VENV_DIR}/bin:/usr/local/bin:/usr/bin"

log() {
  echo "[setup] $*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[setup] missing dependency: $1" >&2
    exit 1
  fi
}

log "checking required commands (python3, node, npm, npx)"
require_cmd python3
require_cmd node
require_cmd npm
require_cmd npx

if [ "${EUID:-$(id -u)}" -ne 0 ]; then
  echo "[setup] root required. Please run with sudo." >&2
  exit 1
fi

log "creating base directories"
mkdir -p "${ROOT_DIR}/_system/db" \
         "${ROOT_DIR}/_system/webapp/storage/uploads" \
         "${ROOT_DIR}/_system/webapp/storage/staging" \
         "${ROOT_DIR}/_system/webapp/storage/output" \
         "${ROOT_DIR}/_system/webapp/tagger_models" \
         "${LOG_DIR}" \
         "${WEBAPP_LOG_DIR}"

log "creating/updating virtualenv at ${VENV_DIR}"
if [ ! -x "${VENV_DIR}/bin/python3" ]; then
  python3 -m venv "${VENV_DIR}"
fi

log "installing Python dependencies"
"${VENV_DIR}/bin/python3" -m pip install --upgrade pip
"${VENV_DIR}/bin/python3" -m pip install -r "${ROOT_DIR}/requirements.txt"
if [ -f "${ROOT_DIR}/trainer/requirements.txt" ]; then
  "${VENV_DIR}/bin/python3" -m pip install -r "${ROOT_DIR}/trainer/requirements.txt"
fi

log "installing webapp dependencies"
pushd "${WEBAPP_DIR}" >/dev/null
npm ci
log "generating Prisma client"
DATABASE_URL="file:${DB_PATH}" npx prisma generate
log "pushing Prisma schema to SQLite"
DATABASE_URL="file:${DB_PATH}" npx prisma db push
popd >/dev/null

log "seeding basic DB data (settings, train profiles, tagger)"
"${VENV_DIR}/bin/python3" "${ROOT_DIR}/scripts/seed_basic_data.py"

log "writing systemd unit files to ${SYSTEMD_DIR}"
mkdir -p "${SYSTEMD_DIR}"

cat > "${SYSTEMD_DIR}/frameforge-db-broker.service" <<SERVICE
[Unit]
Description=FrameForge DB Broker
After=network.target

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${ROOT_DIR}
Environment=DB_BROKER_HOST=127.0.0.1
Environment=DB_BROKER_PORT=8799
Environment=DB_BROKER_DB=${DB_PATH}
Environment=PYTHONUNBUFFERED=1
Environment=VIRTUAL_ENV=${VENV_DIR}
Environment=PATH=${PATH_ENV}
ExecStartPre=/usr/bin/mkdir -p ${WEBAPP_LOG_DIR}
ExecStart=${VENV_DIR}/bin/python3 ${ROOT_DIR}/db_broker.py
Restart=on-failure
StandardOutput=append:${WEBAPP_LOG_DIR}/db-broker.service.log
StandardError=append:${WEBAPP_LOG_DIR}/db-broker.service.log

[Install]
WantedBy=multi-user.target
SERVICE

cat > "${SYSTEMD_DIR}/frameforge-webapp.service" <<SERVICE
[Unit]
Description=FrameForge WebApp
After=network.target frameforge-db-broker.service
Requires=frameforge-db-broker.service
BindsTo=frameforge-db-broker.service

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${WEBAPP_DIR}
Environment=DATABASE_URL=file:${DB_PATH}
Environment=DB_BROKER_URL=${DB_BROKER_URL}
Environment=NODE_ENV=production
Environment=PYTHONUNBUFFERED=1
Environment=VIRTUAL_ENV=${VENV_DIR}
Environment=PATH=${PATH_ENV}
ExecStartPre=/usr/bin/mkdir -p ${WEBAPP_LOG_DIR}
ExecStart=/usr/bin/env node server.js
Restart=on-failure
StandardOutput=append:${WEBAPP_LOG_DIR}/webapp.service.log
StandardError=append:${WEBAPP_LOG_DIR}/webapp.service.log

[Install]
WantedBy=multi-user.target
SERVICE

cat > "${SYSTEMD_DIR}/frameforge-initiator.service" <<SERVICE
[Unit]
Description=FrameForge Initiator
After=network.target frameforge-db-broker.service
Requires=frameforge-db-broker.service
BindsTo=frameforge-db-broker.service

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${ROOT_DIR}
Environment=PYTHONUNBUFFERED=1
Environment=RUN_DB=${DB_PATH}
Environment=DB_BROKER_URL=${DB_BROKER_URL}
Environment=VIRTUAL_ENV=${VENV_DIR}
Environment=PATH=${PATH_ENV}
ExecStartPre=/usr/bin/mkdir -p ${LOG_DIR}
ExecStart=${VENV_DIR}/bin/python3 ${ROOT_DIR}/initiator.py
Restart=always
RestartSec=5
StandardOutput=append:${LOG_DIR}/initiator.service.log
StandardError=append:${LOG_DIR}/initiator.service.log

[Install]
WantedBy=multi-user.target
SERVICE

cat > "${SYSTEMD_DIR}/frameforge-orchestrator.service" <<SERVICE
[Unit]
Description=FrameForge Orchestrator
After=network.target frameforge-db-broker.service
Requires=frameforge-db-broker.service
BindsTo=frameforge-db-broker.service

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${ROOT_DIR}
Environment=PYTHONUNBUFFERED=1
Environment=RUN_DB=${DB_PATH}
Environment=DB_BROKER_URL=${DB_BROKER_URL}
Environment=VIRTUAL_ENV=${VENV_DIR}
Environment=PATH=${PATH_ENV}
ExecStartPre=/usr/bin/mkdir -p ${LOG_DIR}
ExecStart=${VENV_DIR}/bin/python3 ${ROOT_DIR}/orchestrator_worker.py
Restart=always
RestartSec=5
StandardOutput=append:${LOG_DIR}/orchestrator.service.log
StandardError=append:${LOG_DIR}/orchestrator.service.log

[Install]
WantedBy=multi-user.target
SERVICE

cat > "${SYSTEMD_DIR}/frameforge-finisher.service" <<SERVICE
[Unit]
Description=FrameForge Finisher
After=network.target frameforge-db-broker.service
Requires=frameforge-db-broker.service
BindsTo=frameforge-db-broker.service

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${ROOT_DIR}
Environment=PYTHONUNBUFFERED=1
Environment=RUN_DB=${DB_PATH}
Environment=DB_BROKER_URL=${DB_BROKER_URL}
Environment=VIRTUAL_ENV=${VENV_DIR}
Environment=PATH=${PATH_ENV}
ExecStartPre=/usr/bin/mkdir -p ${LOG_DIR}
ExecStart=${VENV_DIR}/bin/python3 ${ROOT_DIR}/finisher.py
Restart=always
RestartSec=5
StandardOutput=append:${LOG_DIR}/finisher.service.log
StandardError=append:${LOG_DIR}/finisher.service.log

[Install]
WantedBy=multi-user.target
SERVICE

log "starting model downloads in background"
"${ROOT_DIR}/scripts/fetch_models.sh" >/dev/null 2>&1 &
MODEL_PID=$!
log "model downloads started (pid ${MODEL_PID})"

log "done. Review unit files in ${SYSTEMD_DIR} then install manually, e.g."
cat <<EOS
  sudo cp ${SYSTEMD_DIR}/frameforge-*.service /etc/systemd/system/
  sudo systemctl daemon-reload
  sudo systemctl enable --now frameforge-db-broker frameforge-webapp frameforge-initiator frameforge-orchestrator frameforge-finisher
EOS
