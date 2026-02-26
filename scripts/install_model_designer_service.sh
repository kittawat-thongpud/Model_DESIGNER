#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="model-designer.service"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_PATH="${SCRIPT_DIR}/${SERVICE_NAME}"
DST_PATH="/etc/systemd/system/${SERVICE_NAME}"

if [[ ! -f "${SRC_PATH}" ]]; then
  echo "Service file not found: ${SRC_PATH}"
  exit 1
fi

echo "Installing ${SERVICE_NAME} -> ${DST_PATH}"
sudo cp "${SRC_PATH}" "${DST_PATH}"
sudo systemctl daemon-reload
sudo systemctl enable --now "${SERVICE_NAME}"

echo "Done. Current status:"
sudo systemctl status "${SERVICE_NAME}" --no-pager -l
