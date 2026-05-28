#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

SKIP_SYSTEM_PACKAGES=0
SKIP_GPU_CHECK=0
for arg in "$@"; do
  case "$arg" in
    --skip-system-packages) SKIP_SYSTEM_PACKAGES=1 ;;
    --skip-gpu-check) SKIP_GPU_CHECK=1 ;;
    *) echo "Unknown argument: $arg"; exit 2 ;;
  esac
done

step() { printf '\n==> %s\n' "$1"; }
warn() { printf 'WARNING: %s\n' "$1" >&2; }

run_as_root() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    return 1
  fi
}

install_system_packages() {
  if [ "$SKIP_SYSTEM_PACKAGES" -eq 1 ]; then
    warn "Skipping system package installation. If Tkinter/OpenCV video fails, install python3-venv, python3-tk, libGL, glib2, and ffmpeg manually."
    return 0
  fi

  step "Installing/checking Linux system packages"

  if command -v apt-get >/dev/null 2>&1; then
    run_as_root apt-get update
    run_as_root apt-get install -y python3 python3-venv python3-pip python3-tk libgl1 libglib2.0-0 ffmpeg || \
      warn "Some apt packages failed. Continue if Python/Tkinter/OpenCV work; otherwise install them manually."
  elif command -v dnf >/dev/null 2>&1; then
    run_as_root dnf install -y python3 python3-pip python3-tkinter mesa-libGL glib2 ffmpeg || \
      warn "Some dnf packages failed. On Fedora/RHEL, ffmpeg may require RPM Fusion."
  elif command -v yum >/dev/null 2>&1; then
    run_as_root yum install -y python3 python3-pip python3-tkinter mesa-libGL glib2 ffmpeg || \
      warn "Some yum packages failed. Install missing packages manually if imports fail."
  elif command -v pacman >/dev/null 2>&1; then
    run_as_root pacman -Sy --needed --noconfirm python python-pip tk libglvnd glib2 ffmpeg || \
      warn "Some pacman packages failed. Install missing packages manually if imports fail."
  elif command -v zypper >/dev/null 2>&1; then
    run_as_root zypper --non-interactive install python3 python3-pip python3-tk tk libGL1 glib2-tools ffmpeg || \
      warn "Some zypper packages failed. Install missing packages manually if imports fail."
  else
    warn "Unknown Linux package manager. The script will continue with Python/pip only."
  fi
}

install_system_packages

step "Checking Python"
if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 was not found. Install Python 3.10+ and rerun this script." >&2
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
printf 'Start the desktop station with:\n  ./start_station_linux.sh\nor:\n  ./.venv/bin/python ./desktop_station_app.py\n\n'
printf 'Note: pip OpenCV is usually CPU-only. If CUDA devices = 0, the app will run on CPU and safely fall back when GPU mode is enabled.\n'
