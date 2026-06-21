# Optional OpenCV station script

This folder keeps the original Python/OpenCV YuNet + SFace script that was used as the behavior reference for the Moodle browser station.

The Moodle plugin does **not** execute this Python file automatically. Moodle is PHP-based, so the installable plugin station uses the React/browser implementation under `react-uploader/` and `uploader/`.

The React station ports the important runtime behavior from this script:

- multiple face detection loop;
- SFace cosine matching threshold profile;
- stable-frame requirement before marking attendance;
- cooldown before marking the same student again;
- unknown-face suppression near known faces;
- short labels on the camera overlay.

Use this Python file only as an optional local desktop reference or for future work if you decide to run an external OpenCV camera service.
