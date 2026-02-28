#!/usr/bin/env bash
# server.sh — manage Model DESIGNER server session via tmux
#
# Usage:
#   ./server.sh start   — start server in background tmux session
#   ./server.sh attach  — attach to running session (Ctrl+B D to detach)
#   ./server.sh logs    — tail server log file
#   ./server.sh stop    — kill server session
#   ./server.sh status  — show if session is running
#   ./server.sh update  — git pull latest code and restart server

SESSION="model-designer"
# Auto-detect APP_DIR from script location; allow override via env var
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${APP_DIR:-${SCRIPT_DIR}}"
LOG_FILE="${APP_DIR}/server.log"
VENV_PYTHON="${APP_DIR}/venv/bin/python3"

# Fall back to system python if venv not present
if [ ! -f "${VENV_PYTHON}" ]; then
    VENV_PYTHON="$(which python3)"
fi

CMD="cd ${APP_DIR}/backend && ${VENV_PYTHON} -m uvicorn app.main:app \
    --host 0.0.0.0 --port 8000 --workers 1 --no-access-log \
    2>&1 | tee ${LOG_FILE}"

case "${1}" in
    start)
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
    update)
        echo "🔄 Pulling latest code from git..."
        git -C "${APP_DIR}" pull
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
