# Facial recognition model test variants with GPU switches

This package contains four runnable variants for testing different facial-recognition models with the same gallery layout.

## Folder layout

Use one folder per person. The files can be numbered: `1.jpg`, `2.jpg`, `3.jpg`, etc.

```text
images/
  GhinescuLucian/
    1.jpg
    2.jpg
    3.jpg
 
```

## Check GPU availability first

Run this in the same environment where you run the scripts:

```powershell
py gpu_check.py
```

Look especially for:

```text
ONNX Runtime providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
TensorFlow GPUs: [PhysicalDevice(...)]
dlib CUDA compiled: True
OpenCV CUDA devices: 1
```

## Important reality check

Not every variant can use the GPU just because the computer has an NVIDIA GPU.

- **Variant 02 InsightFace**: best GPU candidate. It uses `onnxruntime-gpu` and should show `CUDAExecutionProvider`.
- **Variant 04 DeepFace/TensorFlow**: GPU is easiest from WSL2/Linux with `tensorflow[and-cuda]`. Native Windows GPU support is problematic on modern TensorFlow.
- **Variant 03 dlib/face_recognition**: GPU works only if `dlib` was compiled with CUDA. Normal Windows pip installs are usually CPU-only.
- **Variant 01 OpenCV YuNet/SFace**: GPU works only with a custom OpenCV build compiled with CUDA DNN support. The normal `opencv-contrib-python` pip package is usually CPU-only for this path.

## Variant 01 — OpenCV YuNet + SFace

Install CPU:

```powershell
py -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python
py -m pip install -r requirements_opencv_sface.txt
```

Run CPU/auto:

```powershell
py variant_01_opencv_yunet_sface_from_images.py --subjects-dir images --device auto
```

Try CUDA if your OpenCV build supports CUDA DNN:

```powershell
py variant_01_opencv_yunet_sface_from_images.py --subjects-dir images --device cuda
```

## Variant 02 — InsightFace buffalo_l / buffalo_s

This is the most important one to test for the dissertation.

CPU install:

```powershell
py -m pip install -r requirements_insightface_cpu.txt
```

GPU install:

```powershell
py -m pip uninstall -y onnxruntime
py -m pip install -r requirements_insightface_gpu.txt
```

Run GPU:

```powershell
py variant_02_insightface_buffalo.py --subjects-dir images --model-name buffalo_l --device cuda
```

Run a smaller/faster model:

```powershell
py variant_02_insightface_buffalo.py --subjects-dir images --model-name buffalo_s --device cuda
```

## Variant 03 — dlib / face_recognition

Install:

```powershell
py -m pip install -r requirements_dlib_face_recognition.txt
```

Run auto:

```powershell
py variant_03_face_recognition_dlib.py --subjects-dir images --device auto
```

Run CUDA, only if your dlib is compiled with CUDA:

```powershell
py variant_03_face_recognition_dlib.py --subjects-dir images --device cuda
```

If `gpu_check.py` says `dlib CUDA compiled: False`, this variant will not use GPU.

## Variant 04 — DeepFace Facenet512

Install CPU/native Windows:

```powershell
py -m pip install -r requirements_deepface.txt
```

Run CPU:

```powershell
py variant_04_deepface_facenet512.py --subjects-dir images --device cpu
```

Run CUDA, usually from WSL2/Linux:

```bash
python -m pip install -r requirements_deepface_gpu_wsl_linux.txt
python variant_04_deepface_facenet512.py --subjects-dir images --device cuda
```

You can also try a faster detector backend:

```powershell
py variant_04_deepface_facenet512.py --subjects-dir images --detector-backend opencv --process-every-n 3
```

## Practical recommendation

For your real-time classroom attendance tests, prioritize this order:

1. `variant_02_insightface_buffalo.py --device cuda`
2. `variant_01_opencv_yunet_sface_from_images.py` as the fast CPU baseline
3. `variant_04_deepface_facenet512.py` only as a research comparison
4. `variant_03_face_recognition_dlib.py` only as a classic baseline
