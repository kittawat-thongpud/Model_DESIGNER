#!/usr/bin/env bash
# run.sh — Smart Python environment selector
#
# Priority:
#   1. Host python3 has all requirements → use it directly
#   2. venv exists and has all requirements → activate and use
#   3. venv exists but missing deps → pip install -r requirements.txt into venv
#   4. No venv → create venv, install requirements, use it

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ="${SCRIPT_DIR}/requirements.txt"
VENV="${SCRIPT_DIR}/venv"
VENV_PYTHON="${VENV}/bin/python3"
VENV_PIP="${VENV}/bin/pip"

# ── Helper: check if a python interpreter satisfies requirements.txt ─────────
_has_deps() {
    local py="$1"
    local req="$2"
    REQ_FILE="$req" "$py" - <<'EOF' 2>/dev/null
import sys, os, importlib.util, re, pathlib

req_file = pathlib.Path(os.environ["REQ_FILE"])
lines = [l.strip() for l in req_file.read_text().splitlines()
         if l.strip() and not l.startswith('#')]

# Map requirement name → importable name (common mismatches)
_import_map = {
    "pyyaml": "yaml",
    "pillow": "PIL",
    "scikit-learn": "sklearn",
    "scikit-multilearn": "skmultilearn",
    "pycocotools": "pycocotools",
    "opencv-python": "cv2",
    "opencv-python-headless": "cv2",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "torch": "torch",
    "torchvision": "torchvision",
    "ultralytics": "ultralytics",
    "thop": "thop",
    "psutil": "psutil",
    "scipy": "scipy",
    "textual": "textual",
    "gdown": "gdown",
}

missing = []
for line in lines:
    pkg = re.split(r"[><=!;\[]", line)[0].strip().lower()
    import_name = _import_map.get(pkg, pkg.replace("-", "_"))
    if importlib.util.find_spec(import_name) is None:
        missing.append(pkg)

if missing:
    print("MISSING: " + ", ".join(missing), file=sys.stderr)
    sys.exit(1)
sys.exit(0)
EOF
}

# ── 1. Honour pre-set interpreter (e.g. Docker ENV MODEL_DESIGNER_PYTHON) ─────
USE_PYTHON=""

if [ -n "${MODEL_DESIGNER_PYTHON:-}" ] && [ -x "${MODEL_DESIGNER_PYTHON}" ]; then
    echo "✅ Using preset python: ${MODEL_DESIGNER_PYTHON}"
    USE_PYTHON="${MODEL_DESIGNER_PYTHON}"

# ── 2. Check host python ──────────────────────────────────────────────────────
elif HOST_PY="$(which python3 2>/dev/null || true)" && [ -n "$HOST_PY" ] && _has_deps "$HOST_PY" "$REQ"; then
    echo "✅ Using host python: $HOST_PY (all dependencies satisfied)"
    USE_PYTHON="$HOST_PY"
else
    # ── 2/3/4. Fall back to venv ─────────────────────────────────────────────
    if [ ! -d "$VENV" ]; then
        echo "🔧 Creating venv at ${VENV}..."
        python3 -m venv "$VENV"
    fi

    if ! _has_deps "$VENV_PYTHON" "$REQ"; then
        echo "📦 Installing/updating requirements into venv..."
        "$VENV_PIP" install --upgrade pip -q
        "$VENV_PIP" install -r "$REQ"
    else
        echo "✅ Using venv: $VENV_PYTHON (all dependencies satisfied)"
    fi

    # shellcheck disable=SC1091
    . "${VENV}/bin/activate"
    USE_PYTHON="$VENV_PYTHON"
fi

export MODEL_DESIGNER_PYTHON="$USE_PYTHON"
exec "$USE_PYTHON" "${SCRIPT_DIR}/run.py"
