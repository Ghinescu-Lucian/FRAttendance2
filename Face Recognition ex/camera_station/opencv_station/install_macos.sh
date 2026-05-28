#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

SKIP_BREW=0
SKIP_GPU_CHECK=0
for arg in "$@"; do
  case "$arg" in
    --skip-brew) SKIP_BREW=1 ;;
    --skip-gpu-check) SKIP_GPU_CHECK=1 ;;
    *) echo "Unknown argument: $arg"; exit 2 ;;
  esac
done

step() { printf '\n==> %s\n' "$1"; }
warn() { printf 'WARNING: %s\n' "$1" >&2; }

if [ "$(uname -s)" != "Darwin" ]; then
  echo "This installer is for macOS. Use install_linux.sh on Linux." >&2
  exit 1
fi

step "Checking optional Homebrew packages"
if [ "$SKIP_BREW" -eq 0 ] && command -v brew >/dev/null 2>&1; then
  brew list python@3 >/dev/null 2>&1 || brew list python >/dev/null 2>&1 || brew install python || warn "Could not install Homebrew Python. Continuing with the existing python3 if available."
  brew list ffmpeg >/dev/null 2>&1 || brew install ffmpeg || warn "Could not install ffmpeg. Local cameras may still work; some video/RTSP sources may fail."
else
  warn "Homebrew not available or skipped. The script will continue with Python/pip only."
fi

step "Checking Python"
if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 was not found. Install Python 3.10+ from python.org or Homebrew, then rerun this script." >&2
  exit 1
fi
python3 -c "import sys; print(sys.version); assert sys.version_info >= (3, 10), 'Python 3.10 or newer is required'"

step "Creating virtual environment in .venv"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

step "Upgrading pip, setuptools, and wheel"
python -m pip install --upgrade pip setuptools wheel

step "Installing FaceAttendance desktop dependencies"
python -m pip install -r requirements_desktop.txt

step "Creating runtime folders"
mkdir -p images captures unknown_review reviewed_embeddings reports

step "Verifying imports"
python - <<'PY'
import cv2
import numpy
from PIL import Image
import tkinter
print('OK: cv2', cv2.__version__)
PY

if [ "$SKIP_GPU_CHECK" -eq 0 ]; then
  step "Running GPU diagnostic"
  python gpu_check.py
fi

step "Installation complete"
printf 'Start the desktop station with:\n  ./start_station_macos.sh\nor:\n  ./.venv/bin/python ./desktop_station_app.py\n\n'
printf 'Note: CUDA acceleration is normally not available on macOS OpenCV wheels. The app will run on CPU and safely fall back when GPU mode is enabled.\n'
