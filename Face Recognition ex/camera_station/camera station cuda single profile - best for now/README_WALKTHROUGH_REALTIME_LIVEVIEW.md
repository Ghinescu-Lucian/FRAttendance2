# Walk-through realtime + larger live view patch

This patch restores the algorithm-only direction and adds a new profile named `walkthrough_realtime`.

## What changed

- The live camera image now scales up to fill the available preview area. The recognition frame is not upscaled; only the UI image is enlarged.
- New profile: `walkthrough_realtime`.
  - detection every frame (`fast_every = 1`);
  - 960x540 camera input;
  - 640px detector input;
  - SFace CUDA batch remains enabled;
  - known stability reduced from 5 frames to 2 frames, so a normally walking person can be labelled without stopping;
  - unknown drawing and unknown disk writes are disabled by default to reduce lag.
- The desktop app now applies the selected profile before pushing controls, so profile-specific stability values are preserved.

## Recommended settings

- Profile: `walkthrough_realtime`
- Try GPU acceleration: ON
- Draw unknown faces: OFF
- Periodic grid search: OFF
- Max SFace recognitions/frame: 24
- Skip recognition for resolved faces: ON

If you need maximum crowd speed rather than walk-through responsiveness, switch back to `crowd_extreme`.
