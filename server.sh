#!/usr/bin/env bash
# server.sh — manage Model DESIGNER server (auto: systemd service or tmux)
#
# Commands:
#   start             — start server  (service if installed, else tmux)
#   stop              — stop server
#   restart           — restart server
#   update            — git pull + rebuild frontend + restart
#   status            — show running status and active mode
#   attach            — attach to tmux session (tmux mode only)
#   logs              — tail logs  (journalctl in service mode, file in tmux mode)
#   service-install   — install as systemd service (auto-start on boot)
#   service-uninstall — remove systemd service

set -euo pipefail

SESSION="model-designer"
SERVICE_NAME="model-designer"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${APP_DIR:-${SCRIPT_DIR}}"
LOG_FILE="${APP_DIR}/server.log"
VENV_PYTHON="${APP_DIR}/venv/bin/python3"
PORT=8000

export MCP_ALLOWED_HOSTS="${MCP_ALLOWED_HOSTS:-rase*,*.ts.net}"
export MCP_ALLOWED_ORIGINS="${MCP_ALLOWED_ORIGINS:-http://rase*:*,https://rase*:*,http://*.ts.net:*,https://*.ts.net:*}"

# Fall back to system python if venv not present
if [ ! -f "${VENV_PYTHON}" ]; then
    VENV_PYTHON="$(command -v python3 2>/dev/null || echo python3)"
fi

CMD="cd ${APP_DIR} && bash run.sh 2>&1 | tee ${LOG_FILE}"


# ── Helpers ───────────────────────────────────────────────────────────────────

_service_installed() {
    [ -f "${SERVICE_FILE}" ]
}

_port_pids() {
    local pids=""
    if command -v lsof >/dev/null 2>&1; then
        pids="$(lsof -tiTCP:${PORT} -sTCP:LISTEN 2>/dev/null || true)"
    fi

    if [ -z "${pids}" ] && command -v ss >/dev/null 2>&1; then
        pids="$(ss -ltnp "sport = :${PORT}" 2>/dev/null | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | sort -u)"
    fi

    echo "${pids}" | xargs -r echo
}

kill_port() {
    cleanup_training_workers

    local PIDS
    PIDS="$(_port_pids)"
    if [ -n "${PIDS}" ]; then
        echo "⚠️  Port ${PORT} is in use (pid ${PIDS}). Killing..."
        kill -9 ${PIDS} 2>/dev/null || true
        sleep 1
    fi

    PIDS="$(_port_pids)"
    if [ -z "${PIDS}" ]; then
        echo "✅ Port ${PORT} cleared."
    else
        echo "⚠️  Port ${PORT} still in use (pid ${PIDS})."
    fi
}

cleanup_training_workers() {
    local cleaned=0
    local pid_file pid job_id
    shopt -s nullglob
    for pid_file in "${APP_DIR}"/backend/data/jobs/*/worker_process.pid; do
        pid="$(head -n 1 "${pid_file}" 2>/dev/null | tr -dc '0-9' || true)"
        if [ -z "${pid}" ]; then
            rm -f "${pid_file}" "${pid_file%/*}/worker_process.json" 2>/dev/null || true
            continue
        fi

        if ! kill -0 "${pid}" 2>/dev/null; then
            rm -f "${pid_file}" "${pid_file%/*}/worker_process.json" 2>/dev/null || true
            continue
        fi

        job_id="$(basename "$(dirname "${pid_file}")")"
        echo "⚠️  Stale training worker detected (job=${job_id}, pid=${pid}). Killing..."
        kill -TERM "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
        sleep 1
        if kill -0 "${pid}" 2>/dev/null; then
            echo "   Training worker still alive; forcing SIGKILL..."
            kill -KILL "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
            sleep 1
        fi
        if ! kill -0 "${pid}" 2>/dev/null; then
            rm -f "${pid_file}" "${pid_file%/*}/worker_process.json" 2>/dev/null || true
        fi
        cleaned=$((cleaned + 1))
    done
    shopt -u nullglob

    if [ "${cleaned}" -gt 0 ]; then
        echo "✅ Cleaned ${cleaned} stale training worker(s)."
    fi
}

_service_clear_port() {
    # Try user-level cleanup first (same behavior as tmux mode)
    kill_port

    # If another user/service still owns the port, escalate once via sudo.
    local PIDS
    PIDS="$(_port_pids)"
    if [ -n "${PIDS}" ]; then
        echo "⚠️  Port ${PORT} still in use after local cleanup. Forcing release with sudo..."
        sudo fuser -k ${PORT}/tcp >/dev/null 2>&1 || true
        sleep 1
        PIDS="$(_port_pids)"
        if [ -z "${PIDS}" ]; then
            echo "✅ Port ${PORT} cleared with sudo."
        else
            echo "❌ Port ${PORT} is still in use (pid ${PIDS})."
        fi
    fi
}

_build_frontend() {
    if ! command -v node &>/dev/null || ! command -v npm &>/dev/null; then
        echo "📦 Node.js/npm not found — installing via NodeSource (LTS)..."
        curl -fsSL https://deb.nodesource.com/setup_lts.x | bash -
        apt-get install -y nodejs
    fi
    echo "   Node: $(node --version)  npm: $(npm --version)"

    local FRONTEND_DIR="${APP_DIR}/frontend"
    if [ -d "${FRONTEND_DIR}" ]; then
        echo "📦 Installing npm dependencies..."
        npm --prefix "${FRONTEND_DIR}" install
        echo "🔨 Building frontend..."
        if npm --prefix "${FRONTEND_DIR}" run build; then
            echo "✅ Frontend built successfully."
        else
            echo "❌ Frontend build failed. Aborting."
            return 1
        fi
    else
        echo "⚠️  No frontend/ directory found — skipping build."
    fi
}


# ── tmux back-end ─────────────────────────────────────────────────────────────

_tmux_start() {
    kill_port
    if tmux has-session -t "${SESSION}" 2>/dev/null; then
        echo "⚠️  tmux session '${SESSION}' is already running."
        echo "   Use: ./server.sh attach  — to view"
        echo "   Use: ./server.sh stop    — to stop first"
    else
        tmux new-session -d -s "${SESSION}" -x 220 -y 50 "bash -c '${CMD}'"
        sleep 1
        if tmux has-session -t "${SESSION}" 2>/dev/null; then
            echo "✅ Server started in tmux session '${SESSION}'"
            echo "   App: http://localhost:${PORT}"
            echo "   Log: ${LOG_FILE}"
            echo "   Use: ./server.sh attach  — to view live output"
        else
            echo "❌ Session failed to start. Check: cat ${LOG_FILE}"
        fi
    fi
}

_tmux_stop() {
    if tmux has-session -t "${SESSION}" 2>/dev/null; then
        tmux kill-session -t "${SESSION}"
        echo "🛑 tmux session '${SESSION}' stopped."
    else
        echo "⚠️  No tmux session '${SESSION}' running."
    fi
    cleanup_training_workers
    kill_port
}

_tmux_restart() {
    if tmux has-session -t "${SESSION}" 2>/dev/null; then
        echo "� Stopping tmux session '${SESSION}'..."
        tmux kill-session -t "${SESSION}"
        sleep 1
    fi
    cleanup_training_workers
    kill_port
    tmux new-session -d -s "${SESSION}" -x 220 -y 50 "bash -c '${CMD}'"
    sleep 1
    if tmux has-session -t "${SESSION}" 2>/dev/null; then
        echo "✅ Server restarted (tmux session '${SESSION}')"
        echo "   Use: ./server.sh logs  — to verify"
    else
        echo "❌ Failed to restart. Check: cat ${LOG_FILE}"
    fi
}


# ── Commands ──────────────────────────────────────────────────────────────────

case "${1:-}" in

    # ── start ─────────────────────────────────────────────────────────────────
    start)
        if _service_installed; then
            _service_clear_port
            echo "▶️  Starting systemd service '${SERVICE_NAME}'..."
            sudo systemctl start "${SERVICE_NAME}"
            sleep 1
            if systemctl is-active --quiet "${SERVICE_NAME}"; then
                echo "✅ Service '${SERVICE_NAME}' is running."
                echo "   App: http://localhost:${PORT}"
            else
                echo "❌ Service failed to start."
                echo "   Check: journalctl -u ${SERVICE_NAME} -n 50"
            fi
        else
            _tmux_start
        fi
        ;;

    # ── stop ──────────────────────────────────────────────────────────────────
    stop)
        if _service_installed; then
            echo "🛑 Stopping service '${SERVICE_NAME}'..."
            sudo systemctl stop "${SERVICE_NAME}"
            echo "✅ Service '${SERVICE_NAME}' stopped."
            cleanup_training_workers
            kill_port
        else
            _tmux_stop
        fi
        ;;

    # ── restart ───────────────────────────────────────────────────────────────
    restart)
        if _service_installed; then
            _service_clear_port
            echo "🔁 Restarting service '${SERVICE_NAME}'..."
            sudo systemctl restart "${SERVICE_NAME}"
            sleep 1
            if systemctl is-active --quiet "${SERVICE_NAME}"; then
                echo "✅ Service '${SERVICE_NAME}' restarted."
            else
                echo "❌ Restart failed."
                echo "   Check: journalctl -u ${SERVICE_NAME} -n 50"
            fi
        else
            _tmux_restart
        fi
        ;;

    # ── attach ────────────────────────────────────────────────────────────────
    attach)
        if _service_installed; then
            echo "ℹ️  Running as a systemd service — no tmux session."
            echo "   Use: ./server.sh logs  — to follow output"
        else
            if tmux has-session -t "${SESSION}" 2>/dev/null; then
                echo "📎 Attaching to '${SESSION}' (Ctrl+B D to detach)..."
                tmux attach-session -t "${SESSION}"
            else
                echo "❌ No session '${SESSION}' running. Use: ./server.sh start"
            fi
        fi
        ;;

    # ── logs ──────────────────────────────────────────────────────────────────
    logs)
        if _service_installed; then
            echo "📄 Streaming journal for '${SERVICE_NAME}' (Ctrl+C to stop)..."
            journalctl -u "${SERVICE_NAME}" -f --no-pager
        elif [ -f "${LOG_FILE}" ]; then
            echo "� Tailing ${LOG_FILE} (Ctrl+C to stop)..."
            tail -f "${LOG_FILE}"
        else
            echo "❌ No log found: ${LOG_FILE}"
        fi
        ;;

    # ── status ────────────────────────────────────────────────────────────────
    status)
        if _service_installed; then
            echo "── Mode: systemd service ──────────────────────────────────────────────"
            systemctl status "${SERVICE_NAME}" --no-pager -l || true
        else
            echo "── Mode: tmux ─────────────────────────────────────────────────────────"
            if tmux has-session -t "${SESSION}" 2>/dev/null; then
                echo "✅ Session '${SESSION}' is running."
                echo "   Use: ./server.sh attach  — to view"
                echo "   Use: ./server.sh logs    — to tail log"
                echo "   Use: ./server.sh stop    — to stop"
            else
                echo "⛔ Session '${SESSION}' is NOT running."
                echo "   Use: ./server.sh start"
            fi
        fi
        ;;

    # ── update ────────────────────────────────────────────────────────────────
    update)
        echo "🔄 Pulling latest code..."
        git -C "${APP_DIR}" pull

        _build_frontend || exit 1

        echo "🔁 Restarting server..."
        if _service_installed; then
            _service_clear_port
            sudo systemctl restart "${SERVICE_NAME}"
            sleep 1
            if systemctl is-active --quiet "${SERVICE_NAME}"; then
                echo "✅ Service '${SERVICE_NAME}' updated and restarted."
            else
                echo "❌ Restart failed."
                echo "   Check: journalctl -u ${SERVICE_NAME} -n 50"
            fi
        else
            _tmux_restart
        fi
        ;;

    # ── service-install ───────────────────────────────────────────────────────
    service-install)
        if _service_installed; then
            echo "⚠️  Service '${SERVICE_NAME}' is already installed (${SERVICE_FILE})."
            echo "   Use: ./server.sh service-uninstall  — to remove it first"
            exit 1
        fi

        # Determine the user who should own the service
        RUN_USER="${SUDO_USER:-$(whoami)}"
        RUN_GROUP="$(id -gn "${RUN_USER}")"
        RUN_HOME="$(eval echo "~${RUN_USER}")"

        echo "� Installing systemd service '${SERVICE_NAME}' (user: ${RUN_USER})..."

        sudo tee "${SERVICE_FILE}" > /dev/null <<EOF
[Unit]
Description=Model DESIGNER Server
After=network.target

[Service]
Type=simple
User=${RUN_USER}
Group=${RUN_GROUP}
WorkingDirectory=${APP_DIR}
Environment="HOME=${RUN_HOME}"
Environment="APP_DIR=${APP_DIR}"
Environment="MCP_ALLOWED_HOSTS=${MCP_ALLOWED_HOSTS}"
Environment="MCP_ALLOWED_ORIGINS=${MCP_ALLOWED_ORIGINS}"
ExecStart=/bin/bash ${APP_DIR}/run.sh
Restart=on-failure
RestartSec=5s
StandardOutput=journal
StandardError=journal
SyslogIdentifier=${SERVICE_NAME}

[Install]
WantedBy=multi-user.target
EOF

        sudo systemctl daemon-reload
        sudo systemctl enable "${SERVICE_NAME}"
        sudo systemctl start "${SERVICE_NAME}"
        sleep 2

        if systemctl is-active --quiet "${SERVICE_NAME}"; then
            echo "✅ Service '${SERVICE_NAME}' installed, enabled, and running."
            echo "   App:        http://localhost:${PORT}"
            echo "   Logs:       ./server.sh logs"
            echo "   Auto-start: enabled (survives reboot)"
        else
            echo "⚠️  Service installed but failed to start."
            echo "   Check: journalctl -u ${SERVICE_NAME} -n 50"
        fi
        ;;

    # ── service-uninstall ─────────────────────────────────────────────────────
    service-uninstall)
        if ! _service_installed; then
            echo "⚠️  Service '${SERVICE_NAME}' is not installed."
            exit 1
        fi

        echo "�️  Removing systemd service '${SERVICE_NAME}'..."
        sudo systemctl stop    "${SERVICE_NAME}" 2>/dev/null || true
        sudo systemctl disable "${SERVICE_NAME}" 2>/dev/null || true
        sudo rm -f "${SERVICE_FILE}"
        sudo systemctl daemon-reload

        echo "✅ Service '${SERVICE_NAME}' removed."
        echo "   Start in tmux mode with: ./server.sh start"
        ;;

    # ── help ──────────────────────────────────────────────────────────────────
    *)
        echo "Usage: ./server.sh <command>"
        echo ""
        echo "  start             start server  (service if installed, else tmux)"
        echo "  stop              stop server"
        echo "  restart           restart server"
        echo "  update            git pull + rebuild frontend + restart"
        echo "  status            show running status and active mode"
        echo "  attach            attach to tmux session  (tmux mode only)"
        echo "  logs              tail logs  (journal in service mode, file in tmux mode)"
        echo "  service-install   install as systemd service (auto-start on boot)"
        echo "  service-uninstall remove systemd service"
        ;;

esac
