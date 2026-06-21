# Local Python camera station for Moodle Face Attendance

This script runs the OpenCV YuNet + SFace camera station locally and communicates with the Moodle `mod_faceattendance` plugin.

## What it does

1. Downloads registered course embeddings from Moodle using `station_bootstrap.php`.
2. Runs YuNet face detection and SFace recognition locally.
3. Sends known students to Moodle through `station_mark.php`.
4. Sends unknown face descriptors plus temporary JPEG thumbnails to Moodle through `station_unknown.php`.
5. Keeps a local `attendance.csv` as a debug/backup log.

## Configure

Copy the example config:

```powershell
copy station_config.example.json station_config.json
notepad station_config.json
```

Set:

- `moodle_base_url`: your Moodle URL, for example `https://192.168.0.154`
- `cmid`: the id from the activity URL `/mod/faceattendance/view.php?id=18`
- `api_secret`: the Face Attendance API secret from the activity settings
- `verify_tls`: use `false` only for local/self-signed HTTPS certificates
- `camera_index`: usually `0`, `1`, or `2`

## Run

```powershell
py -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python
py -m pip install opencv-contrib-python numpy
py .\main_yunet_sface_many_faces_unknown_fast_short_moodle.py --config .\station_config.json
```

Try another camera:

```powershell
py .\main_yunet_sface_many_faces_unknown_fast_short_moodle.py --config .\station_config.json --camera 1
```

## Moodle requirements

Before running, Moodle must have:

1. An installed `mod_faceattendance` plugin version containing `api/station_bootstrap.php`, `api/station_mark.php`, and `api/station_unknown.php`.
2. At least one registered student embedding in the course.
3. An active attendance session if you want automatic present/late marking.

If there is no active session, the script keeps running and refreshes Moodle periodically.
