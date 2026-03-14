#!/usr/bin/env bash
# server.sh — manage Model DESIGNER server session via tmux
#
# Usage:
#   ./server.sh start   — start server in background tmux session
#   ./server.sh attach  — attach to running session (Ctrl+B D to detach)
#   ./server.sh logs    — tail server log file
#   ./server.sh stop    — kill server session
#   ./server.sh status  — show if session is running
#   ./server.sh restart — restart server without git pull
#   ./server.sh update  — git pull latest code and restart server

SESSION="model-designer"
# Auto-detect APP_DIR from script location; allow override via env var
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${APP_DIR:-${SCRIPT_DIR}}"
LOG_FILE="${APP_DIR}/server.log"
VENV_PYTHON="${APP_DIR}/venv/bin/python3"
PORT=8000

kill_port() {
    PID=$(lsof -ti:${PORT})
    if [ ! -z "$PID" ]; then
        echo "⚠️  Port ${PORT} is in use. Killing process ${PID}..."
        kill -9 $PID
        sleep 1
        echo "✅ Port ${PORT} cleared."
    fi
}
# Fall back to system python if venv not present
if [ ! -f "${VENV_PYTHON}" ]; then
    VENV_PYTHON="$(which python3)"
fi

CMD="cd ${APP_DIR} && bash run.sh 2>&1 | tee ${LOG_FILE}"

case "${1}" in
    start)
        kill_port
        if tmux has-session -t "${SESSION}" 2>/dev/null; then
            echo "⚠️  Session '${SESSION}' is already running."
            echo "   Use: ./server.sh attach  — to view it"
            echo "   Use: ./server.sh stop    — to stop it first"
        else
            tmux new-session -d -s "${SESSION}" -x 220 -y 50 "bash -c '${CMD}'"
            sleep 1
            if tmux has-session -t "${SESSION}" 2>/dev/null; then
                echo "✅ Server started in tmux session '${SESSION}'"
                echo "   App: http://localhost:8000"
                echo "   Log: ${LOG_FILE}"
                echo "   Use: ./server.sh attach  — to view live output"
            else
                echo "❌ Session failed to start. Check: cat ${LOG_FILE}"
            fi
        fi
        ;;
    attach)
        if tmux has-session -t "${SESSION}" 2>/dev/null; then
            echo "📎 Attaching to '${SESSION}' (press Ctrl+B then D to detach)..."
            tmux attach-session -t "${SESSION}"
        else
            echo "❌ No session '${SESSION}' running. Use: ./server.sh start"
        fi
        ;;
    logs)
        if [ -f "${LOG_FILE}" ]; then
            echo "📄 Tailing ${LOG_FILE} (Ctrl+C to stop)..."
            tail -f "${LOG_FILE}"
        else
            echo "❌ Log file not found: ${LOG_FILE}"
        fi
        ;;
    stop)
        if tmux has-session -t "${SESSION}" 2>/dev/null; then
            tmux kill-session -t "${SESSION}"
            echo "🛑 Session '${SESSION}' stopped."
        else
            echo "⚠️  No session '${SESSION}' found."
        fi
        ;;
    status)
        if tmux has-session -t "${SESSION}" 2>/dev/null; then
            echo "✅ Session '${SESSION}' is running."
            echo "   Use: ./server.sh attach  — to view"
            echo "   Use: ./server.sh logs    — to tail log"
            echo "   Use: ./server.sh stop    — to stop"
        else
            echo "⛔ Session '${SESSION}' is NOT running."
            echo "   Use: ./server.sh start   — to start"
        fi
        ;;
    restart)
        if tmux has-session -t "${SESSION}" 2>/dev/null; then
            echo "🔁 Stopping session '${SESSION}'..."
            tmux kill-session -t "${SESSION}"
            sleep 1
        fi
        kill_port
        tmux new-session -d -s "${SESSION}" -x 220 -y 50 "bash -c '${CMD}'"
        sleep 1
        if tmux has-session -t "${SESSION}" 2>/dev/null; then
            echo "✅ Server restarted in tmux session '${SESSION}'"
            echo "   Use: ./server.sh logs  — to verify"
        else
            echo "❌ Failed to restart. Check: cat ${LOG_FILE}"
        fi
        ;;
    update)
        echo "🔄 Pulling latest code from git..."
        git -C "${APP_DIR}" pull

        # ── Install Node.js / npm if missing ──────────────────────────────────
        if ! command -v node &>/dev/null || ! command -v npm &>/dev/null; then
            echo "📦 Node.js/npm not found. Installing via NodeSource (LTS)..."
            curl -fsSL https://deb.nodesource.com/setup_lts.x | bash -
            apt-get install -y nodejs
        fi
        echo "   Node: $(node --version)  npm: $(npm --version)"

        # ── Build frontend ────────────────────────────────────────────────────
        FRONTEND_DIR="${APP_DIR}/frontend"
        if [ -d "${FRONTEND_DIR}" ]; then
            echo "📦 Installing npm dependencies..."
            npm --prefix "${FRONTEND_DIR}" install
            echo "🔨 Building frontend..."
            npm --prefix "${FRONTEND_DIR}" run build
            if [ $? -eq 0 ]; then
                echo "✅ Frontend built successfully."
            else
                echo "❌ Frontend build failed. Aborting restart."
                exit 1
            fi
        else
            echo "⚠️  No frontend/ directory found — skipping build."
        fi

        # ── Restart server ────────────────────────────────────────────────────
        if tmux has-session -t "${SESSION}" 2>/dev/null; then
            echo "🔁 Restarting server..."
            tmux kill-session -t "${SESSION}"
            sleep 1
        fi
        tmux new-session -d -s "${SESSION}" -x 220 -y 50 "bash -c '${CMD}'"
        sleep 1
        if tmux has-session -t "${SESSION}" 2>/dev/null; then
            echo "✅ Server updated and restarted."
            echo "   Use: ./server.sh logs  — to verify"
        else
            echo "❌ Failed to restart. Check: cat ${LOG_FILE}"
        fi
        ;;
    *)
        echo "Usage: ./server.sh {start|attach|logs|stop|status|update}"
        ;;
esac
