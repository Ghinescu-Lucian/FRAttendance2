# FaceAttendance Desktop Station with GPU option, source selector, zoom controls, and unknown review

This package adds a desktop application on top of the existing YuNet + SFace station code.

## New files

- `desktop_station_app.py` - Tkinter desktop station UI.
- `requirements_desktop.txt` - minimal runtime dependencies for the desktop app.
- `run_desktop_station.ps1` - Windows PowerShell launcher.
- `gpu_check.py` - quick OpenCV/CUDA diagnostics.
- `unknown_review/` - created at runtime; stores unknown-track database and pictures.
- `reviewed_embeddings/` - created when you assign a label to an unknown person.
- `reports/` - created when you export a CSV report.

The original CLI station still works. I also updated `moodle_yunet_sface_station.py` and `main_yunet_sface_many_faces_unknown_fast_short_moodle.py` so their `create_detector()` and `create_recognizer()` can try OpenCV DNN CUDA when requested through `FACEATTENDANCE_USE_GPU=true`.

## Install

Use one of the station installer scripts from the `opencv_station` folder. Full details are in `README_INSTALLERS.md`.

### Windows

```powershell
powershell -ExecutionPolicy Bypass -File .\install_windows.ps1
.\start_station_windows.ps1
```

### Linux

```bash
chmod +x install_linux.sh start_station_linux.sh
./install_linux.sh
./start_station_linux.sh
```

### macOS

```bash
chmod +x install_macos.sh start_station_macos.sh install_macos.command
./install_macos.sh
./start_station_macos.sh
```

Manual install still works:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements_desktop.txt
py .\desktop_station_app.py
```

## Using the desktop app

1. Choose the source:
   - **Camera index**: use `0`, `1`, `2`, etc. for USB/webcam devices.
   - **Network stream / video file**: use an RTSP/HTTP stream URL or browse to a local video file.
2. Select the embeddings folder. By default it uses `opencv_station/images`.
3. Choose a profile, for example `fast_short` for speed or `many_faces_unknown` for better unknown review.
4. Enable **Try GPU acceleration**. If your OpenCV build has CUDA, YuNet/SFace will try CUDA. If not, the app falls back to CPU and shows the reason in the status line.
5. Use the zoom controls to digitally zoom before detection and recognition. This is useful for a doorway or classroom entrance.
6. Open **Reports & Unknown Review** to see known and unknown detections.

## Unknown-person workflow

The app clusters unknown faces by their SFace descriptor. For the same unknown track, it saves a picture every 3 detections, with a small time guard to avoid saving dozens of almost identical pictures per second.

In the **Reports & Unknown Review** screen:

1. Select an unknown track such as `UNK-0001`.
2. Preview the last captured picture.
3. Type the real label/name.
4. Click **Assign label**.

The app then:

- marks the unknown track as assigned;
- writes a SFace embedding JSON under `reviewed_embeddings/`;
- adds that label to the running recognizer state, so future detections in the same session can use the new label.

For permanent use across restarts, keep `reviewed_embeddings/` inside the embeddings folder or copy its JSON files into your main embeddings folder.

## GPU notes

The normal `opencv-contrib-python` package from pip usually does **not** include CUDA runtime support. The station will still run, but on CPU. To verify your local machine:

```powershell
py .\gpu_check.py
```

For actual GPU acceleration, `CUDA devices` must be greater than `0`. If it prints `0`, you need an OpenCV build compiled with CUDA and cuDNN support. The UI checkbox is safe: it will not crash just because CUDA is missing; it will fall back to CPU.

## CLI GPU mode

The old CLI station can now try GPU mode too:

```powershell
$env:FACEATTENDANCE_USE_GPU="true"
py .\moodle_yunet_sface_station.py --config .\station_config.json
```

If CUDA is not available, the script prints the fallback reason and continues on CPU.
