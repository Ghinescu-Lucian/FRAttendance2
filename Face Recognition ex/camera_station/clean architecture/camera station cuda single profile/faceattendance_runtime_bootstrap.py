"""Runtime bootstrap for FaceAttendance Station.

This module is intentionally stdlib-only.  It is safe to import before cv2,
Pillow, numpy or onnxruntime.  Its job is to make `py desktop_station_app.py`
self-contained enough for a normal Windows/Linux machine:

* install missing Python packages with pip;
* install NVIDIA CUDA runtime pip packages used by onnxruntime-gpu;
* add NVIDIA DLL folders from site-packages to PATH and the Windows DLL search path.

Set FACEATTENDANCE_DISABLE_AUTO_INSTALL=true to disable pip installs.
Set FACEATTENDANCE_DISABLE_CUDA_DLL_AUTO_PATH=true to disable PATH/DLL setup.
"""

from __future__ import annotations

import importlib
import importlib.metadata as metadata
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

_DLL_HANDLES = []
_BOOTSTRAPPED = False

# Python import name -> pip requirement name.
BASE_IMPORTS = {
    "numpy": "numpy",
}
GUI_IMPORTS = {
    "PIL": "pillow",
}
ORT_IMPORTS = {
    "onnxruntime": "onnxruntime-gpu",
    "onnx": "onnx",
}

NVIDIA_PIP_PACKAGES = [
    "nvidia-cublas-cu12",
    "nvidia-cudnn-cu12",
    "nvidia-cufft-cu12",
    "nvidia-cuda-runtime-cu12",
    "nvidia-curand-cu12",
    "nvidia-cuda-nvrtc-cu12",
    "nvidia-nvjitlink-cu12",
]

# The DLL file names required by onnxruntime-gpu 1.26 CUDA 12.x on Windows.
# Some names can vary slightly by CUDA minor version, so nvrtc/nvjitlink are also
# searched with wildcards below.
REQUIRED_CUDA_DLLS = [
    "cublasLt64_12.dll",
    "cudnn64_9.dll",
    "cufft64_11.dll",
    "cudart64_12.dll",
    "curand64_10.dll",
]
WILDCARD_CUDA_DLLS = [
    "nvrtc*.dll",
    "nvJitLink*.dll",
]


def _truthy_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _run_pip(args: Sequence[str]) -> bool:
    cmd = [sys.executable, "-m", "pip", *args]
    print("[BOOTSTRAP] Running:", " ".join(cmd))
    try:
        subprocess.check_call(cmd)
        return True
    except Exception as exc:
        print(f"[BOOTSTRAP] pip command failed: {exc}")
        return False


def _can_import(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


def _opencv_has_yunet_sface() -> bool:
    try:
        import cv2  # type: ignore
        return hasattr(cv2, "FaceDetectorYN_create") and hasattr(cv2, "FaceRecognizerSF_create")
    except Exception:
        return False


def _missing_import_requirements(include_gui: bool, prefer_gpu: bool) -> List[str]:
    requirements: List[str] = []
    for module_name, requirement in BASE_IMPORTS.items():
        if not _can_import(module_name):
            requirements.append(requirement)

    # OpenCV is special: the station requires YuNet/SFace APIs, so the normal
    # opencv-python package is not enough in many environments.
    if not _opencv_has_yunet_sface():
        requirements.append("opencv-contrib-python")

    if include_gui:
        for module_name, requirement in GUI_IMPORTS.items():
            if not _can_import(module_name):
                requirements.append(requirement)

    if prefer_gpu:
        for module_name, requirement in ORT_IMPORTS.items():
            if not _can_import(module_name):
                requirements.append(requirement)

    # Stable de-duplication while preserving order.
    seen = set()
    unique = []
    for item in requirements:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _distribution_locations(names: Iterable[str]) -> List[Path]:
    locations: List[Path] = []
    for name in names:
        try:
            dist = metadata.distribution(name)
            loc = Path(str(dist.locate_file(""))).resolve()
            if loc.exists():
                locations.append(loc)
        except Exception:
            pass
    return locations


def _site_package_candidates() -> List[Path]:
    candidates: List[Path] = []

    # Most reliable: pip distribution metadata.
    candidates.extend(_distribution_locations(NVIDIA_PIP_PACKAGES + ["onnxruntime-gpu", "onnxruntime"]))

    # Standard Python locations.
    try:
        import site
        for raw in list(getattr(site, "getsitepackages", lambda: [])() or []):
            p = Path(raw).resolve()
            if p.exists():
                candidates.append(p)
        raw_user = getattr(site, "getusersitepackages", lambda: "")()
        if raw_user:
            p = Path(raw_user).resolve()
            if p.exists():
                candidates.append(p)
    except Exception:
        pass

    # Venv and normal Windows install fallback.
    for raw in (getattr(sys, "prefix", ""), getattr(sys, "base_prefix", ""), Path(sys.executable).resolve().parent):
        try:
            p = Path(raw)
            for cand in (p / "Lib" / "site-packages", p / "lib" / "site-packages"):
                if cand.exists():
                    candidates.append(cand.resolve())
        except Exception:
            pass

    seen = set()
    unique: List[Path] = []
    for p in candidates:
        key = str(p).lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def _nvidia_roots() -> List[Path]:
    roots: List[Path] = []
    for site_dir in _site_package_candidates():
        nvidia_root = site_dir / "nvidia"
        if nvidia_root.exists():
            roots.append(nvidia_root)
    seen = set()
    unique: List[Path] = []
    for p in roots:
        key = str(p).lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def _find_nvidia_bin_dirs() -> List[Path]:
    dirs: List[Path] = []
    expected = ["cublas", "cudnn", "cufft", "cuda_runtime", "curand", "cuda_nvrtc", "nvjitlink"]
    for root in _nvidia_roots():
        for name in expected:
            p = root / name / "bin"
            if p.exists():
                dirs.append(p.resolve())
        # Also accept any future nvidia/*/bin folder.
        try:
            for p in root.glob("*/bin"):
                if p.exists():
                    dirs.append(p.resolve())
        except Exception:
            pass

    seen = set()
    unique: List[Path] = []
    for p in dirs:
        key = str(p).lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def add_nvidia_cuda_dll_paths(verbose: bool = False) -> List[str]:
    """Add NVIDIA pip package bin folders to PATH and DLL search path.

    Returns the directories that were found.  On Windows, os.add_dll_directory is
    also used and handles are kept alive globally.
    """
    if _truthy_env("FACEATTENDANCE_DISABLE_CUDA_DLL_AUTO_PATH", False):
        return []

    found_dirs = _find_nvidia_bin_dirs()
    if not found_dirs:
        if verbose:
            print("[BOOTSTRAP] No NVIDIA CUDA pip DLL folders found yet.")
        return []

    current = os.environ.get("PATH", "")
    path_seen = {str(Path(part)).lower() for part in current.split(os.pathsep) if part}
    added_to_path: List[Path] = []
    for folder in found_dirs:
        key = str(folder).lower()
        if key not in path_seen:
            os.environ["PATH"] = str(folder) + os.pathsep + os.environ.get("PATH", "")
            path_seen.add(key)
            added_to_path.append(folder)
        if os.name == "nt":
            try:
                handle = os.add_dll_directory(str(folder))  # type: ignore[attr-defined]
                _DLL_HANDLES.append(handle)
            except Exception:
                pass

    if verbose:
        if added_to_path:
            print("[BOOTSTRAP] Added NVIDIA CUDA DLL paths:")
            for p in added_to_path:
                print(f"  {p}")
        else:
            print("[BOOTSTRAP] NVIDIA CUDA DLL paths were already configured.")
    return [str(p) for p in found_dirs]


def _cuda_dlls_present() -> bool:
    if os.name != "nt":
        # Linux wheels use .so names and dynamic loader behavior is different.
        # Keep this check Windows-specific, where the user hit missing DLL errors.
        return True
    roots = _nvidia_roots()
    if not roots:
        return False
    for dll_name in REQUIRED_CUDA_DLLS:
        found = False
        for root in roots:
            try:
                if next(root.rglob(dll_name), None) is not None:
                    found = True
                    break
            except Exception:
                pass
        if not found:
            return False
    for pattern in WILDCARD_CUDA_DLLS:
        found = False
        for root in roots:
            try:
                if next(root.rglob(pattern), None) is not None:
                    found = True
                    break
            except Exception:
                pass
        if not found:
            return False
    return True


def _install_missing_dependencies(include_gui: bool, prefer_gpu: bool, verbose: bool) -> None:
    if _truthy_env("FACEATTENDANCE_DISABLE_AUTO_INSTALL", False):
        if verbose:
            print("[BOOTSTRAP] Auto-install disabled by FACEATTENDANCE_DISABLE_AUTO_INSTALL.")
        return

    missing = _missing_import_requirements(include_gui=include_gui, prefer_gpu=prefer_gpu)
    if missing:
        _run_pip(["install", "-U", *missing])

    # If OpenCV was imported before installing opencv-contrib, restart is the
    # cleanest option.  Still try to continue because most direct `py app.py`
    # launches start with this bootstrap before cv2 is imported.
    if prefer_gpu:
        add_nvidia_cuda_dll_paths(verbose=False)
        if not _cuda_dlls_present():
            _run_pip([
                "install", "-U", "--only-binary=:all:",
                "onnxruntime-gpu",
                "onnx",
                *NVIDIA_PIP_PACKAGES,
            ])


def _preload_onnxruntime_cuda(verbose: bool = False) -> None:
    try:
        import onnxruntime as ort  # type: ignore
    except Exception:
        return
    try:
        if hasattr(ort, "preload_dlls"):
            # directory="" asks ORT to preload DLLs from NVIDIA pip packages.
            ort.preload_dlls(directory="")
            if verbose:
                print("[BOOTSTRAP] onnxruntime.preload_dlls(directory='') completed.")
    except Exception as exc:
        if verbose:
            print(f"[BOOTSTRAP] onnxruntime.preload_dlls skipped/failed: {exc}")


def ensure_faceattendance_runtime(include_gui: bool = False, prefer_gpu: bool = True, verbose: bool = True) -> None:
    """Install/configure runtime pieces needed by the station.

    This function is idempotent inside a Python process.  It deliberately does
    not create a venv; it installs into the Python interpreter used by `py ...`.
    """
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        add_nvidia_cuda_dll_paths(verbose=False)
        return
    _BOOTSTRAPPED = True

    _install_missing_dependencies(include_gui=include_gui, prefer_gpu=prefer_gpu, verbose=verbose)
    add_nvidia_cuda_dll_paths(verbose=verbose)
    if prefer_gpu:
        _preload_onnxruntime_cuda(verbose=verbose)


if __name__ == "__main__":
    ensure_faceattendance_runtime(include_gui=True, prefer_gpu=True, verbose=True)
    print("[BOOTSTRAP] Done.")
