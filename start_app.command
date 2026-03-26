#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Launching tutoring module app..."

if [[ ! -x ".venv/bin/python" ]]; then
  python3 -m venv .venv
fi

VENV_PYTHON=".venv/bin/python"

"$VENV_PYTHON" -m pip install -r requirements.txt

( sleep 2; open "http://127.0.0.1:8000" ) &
exec "$VENV_PYTHON" -m uvicorn app.main:app --reload --port 8000
