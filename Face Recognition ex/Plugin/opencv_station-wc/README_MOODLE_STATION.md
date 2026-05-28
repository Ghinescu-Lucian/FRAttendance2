# OpenCV YuNet + SFace Moodle camera station

This folder contains the Python camera station that runs outside Moodle and communicates with the `mod_faceattendance` plugin through JSON endpoints.

## Moodle-side requirements

Install this plugin version, then open the Face Attendance activity and copy:

- the Moodle URL, for example `http://localhost/moodle`
- the course module id (`cmid`)
- the API secret configured in the activity settings

Create a scheduled attendance session whose start/end time contains the current time.

## Python setup

```powershell
py -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python
py -m pip install opencv-contrib-python numpy
```

The script uses only Python standard-library HTTP functions, so it does not require `requests`.

## Configure

Open `moodle_yunet_sface_station.py` and change:

```python
MOODLE_BASE_URL = "http://localhost/moodle"
MOODLE_CMID = 12
MOODLE_API_SECRET = "change-this-secret"
CAMERA_INDEX = 0
```

## Run

```powershell
cd moodle\mod\faceattendance\tools\opencv_station
py moodle_yunet_sface_station.py
```

The station will:

1. download registered student embeddings from Moodle;
2. find the active scheduled session;
3. run the original YuNet + SFace multi-face camera loop;
4. mark known students in Moodle after stable detections;
5. send unknown descriptors to Moodle for teacher review.


## Unknown-face review thumbnails

When an unknown face is detected, the station now sends a small JPEG face crop to Moodle together with the SFace descriptor. The teacher can view this temporary image in **Review unknown faces**. After the teacher assigns the unknown face to a student, or ignores it, the plugin deletes the image from Moodle file storage and keeps only the attendance/audit metadata.


## Recognition algorithm profile

The Moodle activity form now includes a **Recognition algorithm profile** field. The external Python station reads this value from `api/station_bootstrap.php` and applies the corresponding YuNet + SFace runtime settings. The included profiles are:

- `fast_short` - fastest default, 960x540, short labels.
- `many_faces_unknown` - balanced many-face detection with unknown boxes.
- `fast_clean` - similar to balanced mode but hides unknown boxes on the live display.
- `high_recall_many_faces` - slower but more aggressive for small/far faces.
- `multi_attendance_zoom` - stronger crop/zoom search.
- `entrance_mode` - doorway/queue profile with entrance-zone focus.

The original experiment files are kept under `tools/opencv_station/examples/` for comparison.
