"""
Check which GPU paths are actually available in the current Python environment.
Run:
  py gpu_check.py
"""

from __future__ import annotations

import sys

print("Python:", sys.executable)
print("Version:", sys.version)
print()

try:
    import cv2
    print("OpenCV:", cv2.__version__)
    cuda_count = cv2.cuda.getCudaEnabledDeviceCount() if hasattr(cv2, "cuda") else 0
    print("OpenCV CUDA devices:", cuda_count)
    print("OpenCV DNN CUDA constants:", hasattr(cv2.dnn, "DNN_BACKEND_CUDA"), hasattr(cv2.dnn, "DNN_TARGET_CUDA"))
except Exception as exc:
    print("OpenCV check failed:", exc)
print()

try:
    import onnxruntime as ort
    print("ONNX Runtime:", ort.__version__)
    print("ONNX Runtime providers:", ort.get_available_providers())
except Exception as exc:
    print("ONNX Runtime check failed:", exc)
print()

try:
    import dlib
    print("dlib:", getattr(dlib, "__version__", "unknown"))
    print("dlib CUDA compiled:", getattr(dlib, "DLIB_USE_CUDA", False))
except Exception as exc:
    print("dlib check failed:", exc)
print()

try:
    import tensorflow as tf
    print("TensorFlow:", tf.__version__)
    print("TensorFlow GPUs:", tf.config.list_physical_devices("GPU"))
except Exception as exc:
    print("TensorFlow check failed:", exc)
