# ONNX Runtime CUDA SFace diagnostics

The warning `Initializer ... appears in graph inputs` is not a crash. It means ONNX Runtime can run the model, but some graph optimizations may be skipped.

This patch adds:

- automatic creation of `models/face_recognition_sface_2021dec_ort_clean.onnx` when the optional `onnx` package is installed;
- explicit logging of the ONNX Runtime providers actually used by the SFace session;
- a benchmark script for pure SFace inference.

## Commands

From `opencv_station`:

```powershell
py -m pip install -U onnx onnxruntime-gpu
py tools\clean_onnx_initializers.py --input models\face_recognition_sface_2021dec.onnx
py tools\benchmark_sface_onnxruntime.py --model models\face_recognition_sface_2021dec_ort_clean.onnx
py .\desktop_station_app.py
```

In the application startup logs, look for:

```text
[DNN] SFace backend: GPU / ONNX Runtime CUDA providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] model=models\face_recognition_sface_2021dec_ort_clean.onnx
```

If the benchmark shows CUDA is fast but the app is still slow, the bottleneck is not SFace inference. It is likely CPU YuNet detection, OpenCV `alignCrop`, camera capture, Tkinter UI drawing, disk writes, or the number of faces processed per frame.
