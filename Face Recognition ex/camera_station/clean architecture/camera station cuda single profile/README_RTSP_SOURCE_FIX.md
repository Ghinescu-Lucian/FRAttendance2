# RTSP/HTTP source stability patch

This build keeps RTSP/HTTP as a normal **Input source** in the Camera Station tab and adds RTSP-specific handling inside the same face-recognition worker used by the local camera and video-file sources.

## How to use

1. Open `desktop_station_app.py` or `desktop_station_app_zoomfix.py`.
2. In **Camera Station > Input source**, choose **RTSP/HTTP stream**.
3. Paste the camera/NVR URL, for example:

   ```text
   rtsp://user:password@192.168.1.108:554/Streaming/Channels/101
   ```

4. Leave **RTSP transport** on `tcp` for most NVR/IP-camera setups.
5. Press **Start**.

## What was fixed

- RTSP is opened through OpenCV's FFmpeg backend first.
- RTSP capture forces a selectable transport: `tcp` or `udp`.
- The default is `tcp`, which is usually more stable on Windows and with NVRs.
- The OpenCV/FFmpeg input buffer is reduced to avoid old delayed frames.
- Open/read timeouts are set when the OpenCV build exposes them.
- If the stream stops returning frames, the worker releases the capture and reconnects automatically.
- Video-file playback remains separate from live RTSP/HTTP streams, so pause/speed controls only affect local video files.

## Notes

If the password contains special URL characters such as `@`, `#`, `?`, `/`, or `:`, URL-encode them before pasting the RTSP URL. For example, `@` becomes `%40`.

Try `udp` only when the TCP stream opens but feels too delayed. UDP can have lower latency, but it is more sensitive to packet loss and Wi-Fi problems.
