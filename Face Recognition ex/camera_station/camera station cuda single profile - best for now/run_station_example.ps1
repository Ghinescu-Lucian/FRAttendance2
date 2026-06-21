# Run from: moodle/mod/faceattendance/tools/opencv_station
py -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python
py -m pip install opencv-contrib-python numpy

Copy-Item .\station_config.example.json .\station_config.json -ErrorAction SilentlyContinue
notepad .\station_config.json

py .\main_yunet_sface_many_faces_unknown_fast_short_moodle.py --config .\station_config.json
