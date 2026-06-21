# GPU low usage after ONNX Runtime CUDA

The SFace benchmark keeps the GPU continuously busy. The desktop station does not:
it runs YuNet detection on CPU, captures frames, aligns crops, updates Tkinter, writes
unknown records, and only sends a small SFace batch every few frames. Therefore low
GPU utilization in the app is expected even when SFace uses CUDA correctly.

This patch adds:

- automatic NVIDIA CUDA DLL path setup for ONNX Runtime on Windows;
- a new `crowd_extreme` profile for CPU-detector bottlenecks;
- throttled desktop output/report refresh to reduce Tkinter overhead;
- per-stage timings in the UI: camera read, YuNet detect, SFace total, SFace align,
  SFace GPU inference, tracking, drawing, total loop.

Recommended diagnostic profile:

```text
Profile: crowd_extreme
Try GPU acceleration: ON
Draw unknown faces: OFF
Stop repeated unknown faces: ON
Max SFace recognitions/frame: 24
```

Interpretation:

- If `det` is high, YuNet CPU detector is the bottleneck.
- If `align` is high but `infer` is low, crop/alignment CPU work is the bottleneck.
- If `track` or report refresh is high, unknown registry/UI is the bottleneck.
- If `read` is high, the camera/driver/source is the bottleneck.
- If `draw` is high, rendering too many boxes/UI preview is the bottleneck.
