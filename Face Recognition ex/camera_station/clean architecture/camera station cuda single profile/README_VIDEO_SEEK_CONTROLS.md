# Video seek controls

This update adds forward/back seeking for video-file recognition.

## Main Camera Station window

When **Input source = Video file** and the station is running, the top bar enables these buttons:

- `-60s` — jump back one minute
- `-10s` — jump back ten seconds
- `+10s` — jump forward ten seconds
- `+60s` — jump forward one minute

The buttons work together with the existing controls:

- Start / Stop
- Pause video / Resume video
- Video speed

Seeking is only enabled for local video files. It is disabled for live camera, RTSP and HTTP streams.

## Separate video window

The optional **Open video window** also has the same seek controls:

- `-60s`
- `-10s`
- `+10s`
- `+60s`

## Implementation notes

The station worker now handles seek actions through a request token. This prevents the same seek command from being executed repeatedly when the UI refreshes or when `push_controls()` runs often.

The worker uses OpenCV seeking with `CAP_PROP_POS_FRAMES`, with `CAP_PROP_POS_MSEC` as fallback.

When seeking while the video is paused, the worker jumps to the requested position and processes/displays one frame, then remains paused.

After a seek, face candidates are cleared and the next frame forces a fresh detection pass so old bounding boxes from the previous video position are not reused.
