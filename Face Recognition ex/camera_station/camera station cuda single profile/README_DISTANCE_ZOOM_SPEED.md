# Distance zoom + speed patch

This patch adds two new profiles for cameras placed far from the subject:

- `distance_zoom_turbo` - default in this patch. Uses 1280x720 capture, 3.2x ROI zoom, lower detector input, throttled UI, no Unknown drawing, and no Unknown disk registry updates.
- `distance_zoom_fast` - higher detail mode. Uses 1920x1080 capture, 2.5x ROI zoom, and a larger detector input. Use it if faces are still too small with `distance_zoom_turbo`.

Why this helps:

- For far faces, using only 640x360 capture and then zooming digitally does not create new face detail.
- These profiles capture more real pixels from the camera, then crop only the region of interest and run detection on that zoomed region.
- This avoids the old slow grid/multi-zoom search while making distant faces larger for YuNet and SFace.

Desktop controls added:

- `Wide` - disables ROI zoom.
- `Far 2.5x` - enables manual ROI zoom at 2.5x.
- `Far 4x` - enables manual ROI zoom at 4x.

Recommended test:

1. Select profile `distance_zoom_turbo`.
2. Keep `Try GPU acceleration` enabled.
3. Keep `Periodic grid search` disabled.
4. Keep `Draw unknown faces` disabled for maximum speed.
5. Use the arrow buttons to place the zoom crop on the door/classroom region.
6. If faces are still too small, press `Far 4x` or select `distance_zoom_fast`.

If GPU usage remains low after this patch, the bottleneck is most likely still CPU-side YuNet detection, camera capture, or Tkinter drawing. The benchmark can keep the GPU busy because it feeds SFace with constant batches; the live app has camera/detector/UI work between GPU calls.
