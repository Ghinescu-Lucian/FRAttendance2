#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [ ! -x .venv/bin/python ]; then
  echo "The virtual environment was not found. Running installer first..."
  ./install_macos.sh
fi

exec .venv/bin/python ./desktop_station_app.py
