# RTSP/H.264 corruption fix

This patch changes the desktop station RTSP defaults to be more stable with HiLook/Hikvision streams:

- forces RTSP over TCP instead of UDP;
- enables low-delay options;
- discards corrupt packets instead of letting the decoder spend time on them;
- lowers the default processing width for network streams from 960 to 640 pixels.

Use the clean RTSP URL, without `?tcp` at the end:

```text
rtsp://admin:PASSWORD@IP_CAMERA:554/Streaming/channels/102
```

Recommended camera/NVR sub-stream settings:

```text
Encoding: H.264
H.264 profile: Baseline or Main
Smart codec / H.264+ / H.265+: OFF
Resolution: 640x360 or 640x480
FPS: 10 or 15
Bitrate type: CBR
Bitrate: 512-1024 kbps
I-frame interval: same as FPS, e.g. 10 or 15
SVC: OFF if available
```

Test raw stream FPS:

```powershell
py .\rtsp_fps_test.py "rtsp://admin:PASSWORD@IP_CAMERA:554/Streaming/channels/102"
```
