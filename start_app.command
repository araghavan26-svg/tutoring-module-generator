#!/bin/bash
set -e

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install -r requirements.txt

( sleep 2; open "http://127.0.0.1:8000" ) &
exec uvicorn app.main:app --reload --port 8000
