# Video Recognition Support

This build supports video recognition in two ways.

## Recommended: use video as the main input source

1. Start the desktop app:

   ```powershell
   py .\desktop_station_app.py
   ```

2. In **Camera Station > Input source**, select **Video file**.
3. Click **Browse file** and choose a local video.
4. Press **Start**.
5. Use the top-bar controls:
   - **Pause video** / **Resume video**
   - **Video speed**: 0.25x, 0.5x, 1x, 1.5x, 2x, 4x
   - **Stop**

This uses the same FR pipeline as the camera/RTSP source.

## Optional: separate video window

The **Open video window** button is still available if you want a separate window. It reuses the same station worker, embeddings, profile, zoom, GPU/Moodle settings and reports.
