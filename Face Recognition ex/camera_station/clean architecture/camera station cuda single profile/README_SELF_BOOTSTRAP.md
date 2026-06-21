# FaceAttendance runtime self-bootstrap

This patch makes the station configure its own Python runtime when launched with:

```powershell
py .\desktop_station_app.py
```

It adds `faceattendance_runtime_bootstrap.py` and calls it before importing OpenCV,
Pillow or ONNX Runtime.

## What it does automatically

- Installs missing Python packages with pip, unless disabled:
  - `numpy`
  - `opencv-contrib-python`
  - `pillow`
  - `onnxruntime-gpu`
  - `onnx`
  - NVIDIA CUDA runtime pip packages needed by ONNX Runtime GPU
- Finds the installed NVIDIA folders under `Lib\site-packages\nvidia\...\bin`.
- Adds those folders to `PATH` and Windows DLL search path with `os.add_dll_directory`.
- Calls `onnxruntime.preload_dlls(directory="")` when available.

The goal is that CUDA DLL errors like this disappear when the app is run directly:

```text
cublasLt64_12.dll is missing
cufft64_11.dll is missing
```

## Disable automatic installs

```powershell
$env:FACEATTENDANCE_DISABLE_AUTO_INSTALL="true"
py .\desktop_station_app.py
```

## Disable automatic CUDA DLL path setup

```powershell
$env:FACEATTENDANCE_DISABLE_CUDA_DLL_AUTO_PATH="true"
py .\desktop_station_app.py
```

## Expected successful log

```text
[DNN] SFace backend: GPU / ONNX Runtime CUDA providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
```
