# Crowd performance patch

This patch targets the slowdown that appears when many faces are visible at the same time.

## Main changes

1. Vectorized known-face matching
   - The old code compared each detected face against every stored embedding with nested Python loops.
   - The new code builds a NumPy matrix of known embeddings and uses one matrix dot product per detected face.

2. Skip SFace for already resolved faces
   - After a known face is confirmed and hidden, later detections that overlap that face reuse the existing confirmed state.
   - After an Unknown face reaches the configured repeated-detection threshold and is hidden, later detections also reuse its hidden track.
   - This means YuNet can still detect the face, but SFace is not run again for faces that no longer need operator attention.

3. Crowd Fast profile
   - New profile: `crowd_fast`.
   - Intended for roughly 20-40 people in the frame.
   - Uses 960x540, smaller detection input, grid search disabled, and max 18 SFace recognitions per frame.

4. Desktop UI controls
   - Added `Skip recognition for resolved faces`.
   - Added `Max SFace recognitions/frame (0 = all)`.

## Recommended settings for 35 people

In `station_config.json`:

```json
"profile": "crowd_fast",
"skip_resolved_recognition": true,
"max_recognitions_per_frame": 18,
"hide_confirmed_known_faces": true,
"confirmed_similarity_threshold": 0.52,
"confirmed_stable_frames": 5,
"hide_confirmed_unknown_faces": true,
"confirmed_unknown_frames": 10
```

If recognition feels too slow to finish the class, increase `max_recognitions_per_frame` to 24 or set it to 0 for unlimited.
If FPS is still too low, reduce it to 12.

## GPU note

Low GPU usage is expected with OpenCV YuNet/SFace because the recognizer is called per face and much of the frame pipeline is CPU-side. This patch reduces repeated SFace calls instead of relying on GPU utilization.
