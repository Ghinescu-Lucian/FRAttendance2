import cv2

print("Python OpenCV:", cv2.__version__)
print("Has cv2.cuda:", hasattr(cv2, "cuda"))
try:
    print("CUDA devices:", cv2.cuda.getCudaEnabledDeviceCount() if hasattr(cv2, "cuda") else 0)
except Exception as exc:
    print("CUDA device check failed:", exc)

print("DNN_BACKEND_CUDA constant:", hasattr(cv2.dnn, "DNN_BACKEND_CUDA"))
print("DNN_TARGET_CUDA constant:", hasattr(cv2.dnn, "DNN_TARGET_CUDA"))

try:
    build_lines = cv2.getBuildInformation().splitlines()
    for line in build_lines:
        if "NVIDIA CUDA" in line or "cuDNN" in line or "DNN" in line or "FFMPEG" in line:
            print(line)
except Exception:
    pass

print("\nInterpretation:")
print("- CUDA devices > 0 means the desktop station can try OpenCV DNN CUDA.")
print("- CUDA devices = 0 means it will run correctly, but on CPU, unless you install/build OpenCV with CUDA support.")
