#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${DEST:-/opt/FrameForge}"
APP_USER="${FRAMEFORGE_USER:-${SUDO_USER:-$USER}}"

if [[ $EUID -ne 0 ]]; then
  echo "Please run as root (sudo) so systemd units and /opt writes work."
  exit 1
fi

echo "[install] syncing files to ${DEST}"
rsync -a --delete \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude 'webapp/node_modules' \
  "${SRC_DIR}/" "${DEST}/"

mkdir -p "${DEST}/webapp/logs" "${DEST}/webapp/storage/uploads" "${DEST}/webapp/storage/output"
chown -R "${APP_USER}:${APP_USER}" "${DEST}"

echo "[install] python venv + deps"
python3 -m venv "${DEST}/.venv"
"${DEST}/.venv/bin/pip" install --upgrade pip
"${DEST}/.venv/bin/pip" install -r "${DEST}/requirements.txt"

echo "[install] npm install + prisma"
runuser -u "${APP_USER}" -- bash -lc "cd '${DEST}/webapp' && npm ci"
runuser -u "${APP_USER}" -- bash -lc "cd '${DEST}/webapp' && DATABASE_URL=file:./storage/db.sqlite npx prisma generate && DATABASE_URL=file:./storage/db.sqlite npx prisma db push"

echo "[install] seeding AutoChar presets"
runuser -u "${APP_USER}" -- bash -lc "cd '${DEST}' && DATABASE_URL=file:./webapp/storage/db.sqlite '${DEST}/.venv/bin/python' scripts/seed_autochar.py"

WEBAPP_UNIT="/etc/systemd/system/frameforge-webapp.service"
QUEUE_UNIT="/etc/systemd/system/frameforge-queue.service"

echo "[install] writing systemd units"
cat > "${WEBAPP_UNIT}" <<SERVICE
[Unit]
Description=FrameForge WebApp
After=network.target

[Service]
Type=simple
User=${APP_USER}
WorkingDirectory=${DEST}/webapp
Environment=DATABASE_URL=file:./storage/db.sqlite
Environment=NODE_ENV=production
Environment=VIRTUAL_ENV=${DEST}/.venv
Environment=PATH=${DEST}/.venv/bin:/usr/local/bin:/usr/bin
ExecStartPre=/usr/bin/mkdir -p ${DEST}/webapp/logs
ExecStart=/usr/bin/env node server.js
Restart=on-failure
StandardOutput=append:${DEST}/webapp/logs/webapp.service.log
StandardError=append:${DEST}/webapp/logs/webapp.service.log

[Install]
WantedBy=multi-user.target
SERVICE

cat > "${QUEUE_UNIT}" <<SERVICE
[Unit]
Description=FrameForge queue watcher (auto pipeline trigger)
After=network.target

[Service]
Type=simple
User=${APP_USER}
WorkingDirectory=${DEST}
Environment=PYTHONUNBUFFERED=1
Environment=VIRTUAL_ENV=${DEST}/.venv
Environment=PATH=${DEST}/.venv/bin:/usr/local/bin:/usr/bin
ExecStartPre=/usr/bin/mkdir -p ${DEST}/webapp/logs
ExecStart=${DEST}/.venv/bin/python ${DEST}/queue_watcher.py
Restart=always
RestartSec=5
StandardOutput=append:${DEST}/webapp/logs/queue.service.log
StandardError=append:${DEST}/webapp/logs/queue.service.log

[Install]
WantedBy=multi-user.target
SERVICE

echo "[install] enabling services"
systemctl daemon-reload
systemctl enable --now frameforge-webapp.service
systemctl enable --now frameforge-queue.service

echo "[install] done. Webapp logs: ${DEST}/webapp/logs"
