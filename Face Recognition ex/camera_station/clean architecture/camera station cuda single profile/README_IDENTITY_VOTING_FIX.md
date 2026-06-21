# Identity voting fix for false Known-person rows

## Problem fixed

When the embedding database is large, SFace can occasionally produce one-frame false matches. The old station accepted every frame-level match immediately, so `Reports & Unknown Review > Known persons` could show many people even when only one real person was in the camera.

## What changed

The station now uses a per-physical-face identity voter before a known identity is allowed to reach:

- `Known persons` report
- attendance marking
- known detection counters
- confirmed/hidden known-face state

A candidate label is now treated as `CHECK <name>` until the same tracked face receives enough consistent votes.

Default confirmation rule:

- same tracked face must collect at least 4 usable votes;
- the winning name must have at least 60% of the recent vote window;
- the winning name must beat the second name by at least 2 votes;
- the average similarity must be at least 0.40;
- ambiguous matches must have at least a small margin over the second-best person.

After confirmation, transient wrong labels on the same face are overridden by the locked identity instead of creating new report rows.

## Files changed

- `moodle_yunet_sface_station.py`
- `desktop_station_app.py`
- `desktop_station_app_zoomfix.py`

## Useful tuning environment variables

Normally you should not need these. They are available if your camera/lighting requires adjustment.

```powershell
# Disable the new voting gate entirely
$env:FACEATTENDANCE_IDENTITY_VOTING="false"

# Make confirmation stricter
$env:FACEATTENDANCE_IDENTITY_MIN_VOTES="5"
$env:FACEATTENDANCE_IDENTITY_MIN_RATIO="0.70"
$env:FACEATTENDANCE_IDENTITY_MIN_SIMILARITY="0.43"

# Make confirmation faster, but less strict
$env:FACEATTENDANCE_IDENTITY_MIN_VOTES="3"
$env:FACEATTENDANCE_IDENTITY_MIN_RATIO="0.55"
$env:FACEATTENDANCE_IDENTITY_MIN_SIMILARITY="0.38"
```

Recommended first test: keep the defaults, stand alone in front of the camera for 10-20 seconds, and check that only your real identity remains in `Known persons`.
