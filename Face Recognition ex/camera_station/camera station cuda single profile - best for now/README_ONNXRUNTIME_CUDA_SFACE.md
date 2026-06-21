# ONNX Runtime CUDA SFace patch

This patch keeps the existing OpenCV YuNet detector, but moves the expensive SFace embedding extraction to ONNX Runtime CUDA when available.

Why this helps:

- Your `nvidia-smi` result shows the RTX 4070 is available.
- Your `cv2.cuda.getCudaEnabledDeviceCount()` result shows OpenCV sees `0` CUDA devices, so OpenCV DNN is CPU-only.
- ONNX Runtime CUDA can still use the GPU even when pip OpenCV cannot.
- This patch also batches all SFace crops from the same frame into one ONNX Runtime call, which is much better for GPU utilization than calling SFace one face at a time.

## Install / verify

In PowerShell:

```powershell
py -m pip uninstall -y onnxruntime onnxruntime-gpu
py -m pip install -U onnxruntime-gpu
py -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
```

Good result:

```text
['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

Also watch GPU usage live:

```powershell
nvidia-smi -l 1
```

## Run

```powershell
py .\desktop_station_app.py
```

Keep `Try GPU acceleration (ONNX Runtime CUDA for SFace)` checked. In the camera status line you should see something like:

```text
SFace 8/frame/1 batch | Detector: CPU fallback ...; Recognizer: GPU / ONNX Runtime CUDA
```

## Fallback behavior

If ONNX Runtime CUDA is not available, the app falls back to the previous OpenCV path and prints the reason in the terminal.

Advanced debug switches:

```powershell
$env:FACEATTENDANCE_SFACE_BACKEND="onnxruntime_cuda"  # force ORT and fail if unavailable
$env:FACEATTENDANCE_SFACE_BACKEND="opencv"            # force old OpenCV SFace
$env:FACEATTENDANCE_SFACE_ORT_USE_RGB="true"          # debug only if embeddings mismatch
```
