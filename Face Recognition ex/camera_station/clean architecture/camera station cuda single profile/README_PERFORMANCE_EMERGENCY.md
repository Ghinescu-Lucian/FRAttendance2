# Performance emergency patch

This patch targets the case where the station becomes very slow with 25-40+ faces.

Main changes:

1. Adds profile `crowd_turbo`:
   - camera 640x360
   - recognition input 384 px
   - detects every 3rd frame
   - no grid search
   - max 8 SFace recognitions per frame

2. Fixes a desktop-app performance bug:
   - the previous desktop worker overwrote the selected profile camera size with 1280x720.
   - now the selected profile resolution is respected.

3. Builds the known-embedding matrix once:
   - avoids rebuilding/rechecking the known embedding index for every face.

4. Reduces unknown-review disk writes:
   - unknown crops are saved every 10 detections instead of every 3.
   - unknown registry JSON is not rewritten for every unknown face in every frame.
   - at most 2 unknown tracks are updated in the review DB per frame.

5. Adds UI status: `SFace/frame`.
   - If this value is high, recognition is the bottleneck.
   - If it is low but FPS is still bad, capture/UI/detection/disk/CPU is the bottleneck.

Recommended first test:
- Profile: `crowd_turbo`
- Grid search: OFF
- Max SFace recognitions/frame: 8
- Draw unknown faces: ON only while reviewing unknowns; OFF for maximum attendance speed.
- Stop known after detections: 150
