# RTSP/network stream freeze + digital zoom fix

This patch changes `desktop_station_app.py` only.

## What it fixes

1. Network stream freezes no longer block the whole station loop.
   RTSP/HTTP/video-file reading is moved to a resilient background reader that reconnects when frames stop arriving.

2. Zoom/pan now behaves correctly for network streams.
   The camera/NVR does not receive hardware zoom/pan commands. The station applies digital zoom by cropping the frame and resizing it before face detection and display.

3. The zoom checkbox is now practical.
   If you enable zoom while the factor is still `1.0`, the app automatically sets it to `1.5`, otherwise nothing visible would happen.

4. Pan arrows now enable digital zoom automatically.
   If you click an arrow while zoom is still `1.0`, the app switches to `1.5` and pans the cropped region.

## How to apply

Copy the patched file over your existing file:

```powershell
copy .\desktop_station_app.py "C:\path\to\opencv_station\desktop_station_app.py"
```

or extract this archive over your existing `opencv_station` folder.

## Recommended RTSP source

Use the camera/NVR sub-stream first:

```text
rtsp://admin:YOUR_PASSWORD@CAMERA_IP:554/Streaming/channels/102
```

For an NVR channel 2 sub-stream:

```text
rtsp://admin:YOUR_PASSWORD@NVR_IP:554/Streaming/channels/202
```

Recommended camera stream settings:

- Codec: H.264
- Resolution: 640x360 or 640x480 first
- FPS: 10-15
- Bitrate: 512-1024 kbps

Avoid H.265 initially because some OpenCV/FFmpeg builds are unstable or slow with it.

## Important

This is digital zoom, not optical PTZ. It works on any frame source, including RTSP, because it crops the received image locally.
