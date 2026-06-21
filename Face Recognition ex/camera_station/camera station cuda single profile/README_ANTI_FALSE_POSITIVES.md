# Anti false-positive changes for 200+ embeddings

This version makes the OpenCV YuNet + SFace station more conservative when the enrolled embedding pool grows.

## What changed

1. **Stricter SFace threshold**
   - Default changed from `0.36` to `0.45`.
   - Override with `FACEATTENDANCE_SFACE_THRESHOLD` or `sface_similarity_threshold` in `station_config.json`.

2. **Top1 / Top2 margin check**
   - A person is accepted only if the best match is clearly better than the second-best person.
   - Default margin: `0.06`.
   - Override with `FACEATTENDANCE_SFACE_MARGIN` or `sface_margin_threshold` in `station_config.json`.

3. **Minimum face size check**
   - Faces smaller than `80px` on their shortest side are not recognized, because far/small faces produce unstable embeddings.
   - Override with `FACEATTENDANCE_MIN_FACE_SIZE` or `min_recognition_face_size` in `station_config.json`.

4. **Per-person best score before top2 comparison**
   - The matcher now computes the best score for each person first.
   - This avoids treating multiple embeddings of the same person as the second-best conflict.

5. **Spatial stability reset**
   - The stable-frame counter resets if the same label appears far away from the previous box.
   - This prevents collecting confirmations from different physical faces.

## Recommended tuning

Start with:

```json
"sface_similarity_threshold": 0.45,
"sface_margin_threshold": 0.06,
"min_recognition_face_size": 80
```

If too many known students become `Unknown`, reduce gradually:

```json
"sface_similarity_threshold": 0.42,
"sface_margin_threshold": 0.04
```

If wrong labels still appear, increase gradually:

```json
"sface_similarity_threshold": 0.48,
"sface_margin_threshold": 0.08,
"min_recognition_face_size": 100
```

## Files modified

- `moodle_yunet_sface_station.py`
- `main_yunet_sface_many_faces_unknown_fast_short_moodle.py`
- `main_yunet_sface_many_faces_unknown_fast_short.py`
- `desktop_station_app.py`
- `desktop_station_app_zoomfix.py`
- `station_config.json`
