# Algorithm-only performance restore patch

This patch reverts the distance-zoom UI/profile changes and keeps the performance-oriented algorithmic changes:

- SFace on ONNX Runtime CUDA with dynamic batch support.
- Automatic NVIDIA CUDA DLL path setup from Python site-packages.
- `crowd_extreme` performance profile.
- Reduced UI/report refresh overhead.
- Reduced Unknown registry/disk update churn.
- Profiling timings for read/detect/align/infer/track/draw/loop.
- Unknown review controls, sorting dropdowns, multi-select delete, and no-photo cleanup.

What is reverted:

- Default profile is no longer `distance_zoom_turbo`.
- Manual zoom is no longer automatically enabled.
- `Wide`, `Far 2.5x`, and `Far 4x` UI preset buttons are removed.
- `distance_zoom_fast` and `distance_zoom_turbo` profiles are removed from the code/config.

Recommended test settings:

- Profile: `crowd_extreme`
- Try GPU acceleration: ON
- Draw unknown faces: OFF
- Skip recognition for resolved faces: ON
- Stop repeated unknown faces: ON
- Max SFace recognitions/frame: 16-24
- Periodic grid search: OFF
