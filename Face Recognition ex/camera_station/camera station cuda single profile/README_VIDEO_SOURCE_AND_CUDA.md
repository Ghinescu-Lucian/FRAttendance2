# Video file as an input source + CUDA note

This build lets a local video file be used directly from the main **Camera Station** tab, beside the normal camera/RTSP choices.

## Use a video file like a source

1. Start the app:

   ```powershell
   py .\desktop_station_app.py
   ```

2. In **Input source**, select **Video file**.
3. Click **Browse file** and select `.mp4`, `.avi`, `.mkv`, `.mov`, `.m4v`, or `.webm`.
4. Keep the normal recognition settings: embeddings source, Moodle, profile, zoom, unknown review, etc.
5. Press **Start** in the top bar.
6. Use the top-bar video controls:
   - **Pause video** / **Resume video**
   - **Video speed**: `0.25x`, `0.5x`, `1x`, `1.5x`, `2x`, `4x`
   - **Stop** to stop the video worker

The video path is processed by the same `StationWorker` and the same FR pipeline used by the live camera feed.

The separate **Open video window** button is still available, but it is optional. The main recommended path is now **Input source -> Video file**.

## What “no CUDA-enabled OpenCV device detected” means

This message means the detector tried to use OpenCV DNN CUDA, but the current OpenCV runtime did not expose a usable CUDA device to OpenCV.

Common causes:

- You installed the normal `opencv-python` wheel from pip. That package is usually CPU-only for CUDA DNN usage.
- The machine has no NVIDIA GPU.
- The NVIDIA driver/CUDA runtime is missing or not visible to Python.
- OpenCV was not built with CUDA support, even if the computer has an NVIDIA GPU.

This is not a fatal error. The station falls back to CPU detection and continues working. In this project, SFace recognition may still use ONNX Runtime CUDA if ONNX Runtime sees the GPU, while YuNet detection may remain on CPU if OpenCV CUDA is unavailable.

Run this to check what Python sees:

```powershell
py .\gpu_check.py
```

If it prints `OpenCV CUDA devices: 0`, the overlay message is expected.
