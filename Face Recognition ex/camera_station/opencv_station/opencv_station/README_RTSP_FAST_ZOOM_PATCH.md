# RTSP fast + digital zoom patch

This patch replaces `desktop_station_app.py` with a version that fixes two issues:

1. RTSP/network stream reading no longer blocks the recognition loop. The app reads RTSP frames in a background thread and always processes the newest fresh frame.
2. Zoom and pan are forced to work as digital crop + resize before detection and display. Moving the zoom slider above 1x enables zoom automatically; pressing an arrow button also enables 2x zoom automatically.

## How to apply

Copy the patched files over your `opencv_station` folder:

```powershell
copy .\desktop_station_app.py "C:\Users\Ghinescu Lucian\Desktop\Master\DISERTATIE\Cod\Face Recognition ex\camera_station\opencv_station\desktop_station_app.py"
copy .\rtsp_fps_test.py "C:\Users\Ghinescu Lucian\Desktop\Master\DISERTATIE\Cod\Face Recognition ex\camera_station\opencv_station\rtsp_fps_test.py"
```

Then start normally:

```powershell
py .\desktop_station_app.py
```

or:

```powershell
.\start_station_windows.bat
```

## Recommended RTSP URL

Use the sub-stream first:

```text
rtsp://admin:PAROLA@IP_CAMERA:554/Streaming/channels/102
```

For NVR channel 2 sub-stream:

```text
rtsp://admin:PAROLA@IP_NVR:554/Streaming/channels/202
```

## Camera/NVR stream settings

Set the camera sub-stream to:

```text
Codec: H.264
Resolution: 640x360, 640x480, or 1280x720
FPS: 10-15
Bitrate: 512-1024 kbps for 640p, 1024-2048 kbps for 720p
I-frame interval: same as FPS, e.g. 10 or 15
```

Avoid H.265 until everything is stable.

## Raw RTSP FPS test

Run:

```powershell
py .\rtsp_fps_test.py "rtsp://admin:PAROLA@IP_CAMERA:554/Streaming/channels/102"
```

Interpretation:

- If `RAW_RTSP_FPS` is below 8-10 FPS, the camera stream/network/camera settings are the bottleneck.
- If `RAW_RTSP_FPS` is good but the desktop station is slow, the recognition pipeline is the bottleneck. Keep the stream at 640p or 720p and disable periodic grid search.
