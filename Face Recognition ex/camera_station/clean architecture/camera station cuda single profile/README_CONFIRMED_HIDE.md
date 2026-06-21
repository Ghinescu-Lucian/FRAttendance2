# Confirmed / repeated-face hiding patch

This patch adds a "resolved" state for both known people and repeated unknown people in the OpenCV station.

## Behavior for known people

A known face is hidden from the live camera view after both conditions are true:

1. the same person has been stable for `confirmed_stable_frames` frames;
2. the best SFace similarity is at least `confirmed_similarity_threshold`.

Default values:

```json
"hide_confirmed_known_faces": true,
"confirmed_similarity_threshold": 0.52,
"confirmed_stable_frames": 5
```

When a known person becomes confirmed, the program no longer draws the green box/name over that person. The person is still tracked internally, so attendance/proof-photo logic still works.

## Behavior for repeated Unknown people

A repeated Unknown face is now grouped using the SFace descriptor. After the same Unknown track appears enough times, it is treated as already reviewed/seen and hidden from the live view.

Default values:

```json
"hide_confirmed_unknown_faces": true,
"confirmed_unknown_frames": 10,
"unknown_track_match_threshold": 0.50
```

Meaning:

- `confirmed_unknown_frames`: after how many repeated detections the same Unknown face is hidden;
- `unknown_track_match_threshold`: how similar two Unknown descriptors must be to count as the same unknown person;
- lower `unknown_track_match_threshold` groups more aggressively;
- higher `unknown_track_match_threshold` is stricter and safer if multiple unknown people look similar.

For the desktop unknown-review database, the repeated unknown is captured only while it is still unresolved. With the default `10`, it can still collect review crops before being hidden.

## Desktop app

Run normally:

```powershell
py .\desktop_station_app.py
```

The UI now has:

- `Hide confirmed known faces`
- `Confirmed known similarity` slider
- `Stop repeated unknown faces`
- `Stop Unknown after detections` slider

Recommended values:

- Use `10` for `Stop Unknown after detections` as a balanced default.
- Use `5–7` if you want unknown people to disappear faster.
- Use `15–20` if you want more review photos before the unknown is hidden.

## CLI / Moodle station

The new values are read from `station_config.json`:

```json
"hide_confirmed_known_faces": true,
"confirmed_similarity_threshold": 0.52,
"confirmed_stable_frames": 5,
"hide_confirmed_unknown_faces": true,
"confirmed_unknown_frames": 10,
"unknown_track_match_threshold": 0.50
```

You can also override them from the command line:

```powershell
py .\main_yunet_sface_many_faces_unknown_fast_short_moodle.py --config .\station_config.json --confirmed-similarity 0.55 --confirmed-stable-frames 5 --confirmed-unknown-frames 10
```

To keep drawing confirmed known people:

```powershell
py .\main_yunet_sface_many_faces_unknown_fast_short_moodle.py --config .\station_config.json --show-confirmed
```

To keep drawing repeated unknown people:

```powershell
py .\main_yunet_sface_many_faces_unknown_fast_short_moodle.py --config .\station_config.json --show-confirmed-unknown
```

## Files changed

- `desktop_station_app.py`
- `desktop_station_app_zoomfix.py`
- `moodle_yunet_sface_station.py`
- `main_yunet_sface_many_faces_unknown_fast_short_moodle.py`
- `main_yunet_sface_many_faces_unknown_fast_short.py`
- `station_config.json`
