# FaceAttendance Station installers

These scripts prepare a station machine for the desktop app by creating a local Python virtual environment and installing the required Python packages.

They do **not** silently install a CUDA-enabled OpenCV build. The normal `opencv-contrib-python` wheel from pip is usually CPU-only. After installation, each script runs `gpu_check.py` so you can see whether the current station can actually use CUDA.

## Windows

Open PowerShell in this folder and run:

```powershell
powershell -ExecutionPolicy Bypass -File .\install_windows.ps1
```

Or double-click:

```text
install_windows.bat
```

Start the app later with:

```powershell
.\start_station_windows.ps1
```

Or double-click:

```text
start_station_windows.bat
```

What it does:

- checks for Python 3.10+ using `py -3` or `python`;
- creates `.venv`;
- installs `requirements_desktop.txt`;
- creates `images`, `captures`, `unknown_review`, `reviewed_embeddings`, and `reports` folders;
- verifies `cv2`, `numpy`, `PIL`, and `tkinter` imports;
- runs the GPU diagnostic.

## Linux

Open a terminal in this folder and run:

```bash
chmod +x install_linux.sh start_station_linux.sh
./install_linux.sh
```

Start the app later with:

```bash
./start_station_linux.sh
```

The Linux installer tries to install system packages with the available package manager:

- Debian/Ubuntu: `apt-get`
- Fedora/RHEL: `dnf` or `yum`
- Arch: `pacman`
- openSUSE: `zypper`

It installs packages such as Python, venv support, Tkinter, OpenGL runtime libraries, GLib, and ffmpeg. If you do not want it to touch system packages, run:

```bash
./install_linux.sh --skip-system-packages
```

## macOS

Open Terminal in this folder and run:

```bash
chmod +x install_macos.sh start_station_macos.sh install_macos.command
./install_macos.sh
```

Start the app later with:

```bash
./start_station_macos.sh
```

You can also double-click `install_macos.command` from Finder.

If Homebrew is installed, the script tries to install/check Python and ffmpeg. If you do not want it to use Homebrew, run:

```bash
./install_macos.sh --skip-brew
```

## GPU expectations

The desktop app has a **Try GPU acceleration** checkbox. It is safe to enable: if CUDA is unavailable, the app falls back to CPU.

For real CUDA acceleration on Windows/Linux, `gpu_check.py` must show at least one CUDA device and an OpenCV build compiled with CUDA/cuDNN. Installing `opencv-contrib-python` from pip alone usually does not provide this.

## Common fixes

### `ModuleNotFoundError: tkinter`

Install Tkinter for your OS:

- Ubuntu/Debian: `sudo apt-get install python3-tk`
- Fedora: `sudo dnf install python3-tkinter`
- Arch: `sudo pacman -S tk`
- macOS: install Python from python.org or Homebrew and rerun the installer.

### `ImportError: libGL.so.1`

Install OpenGL runtime libraries:

- Ubuntu/Debian: `sudo apt-get install libgl1 libglib2.0-0`
- Fedora: `sudo dnf install mesa-libGL glib2`
- Arch: `sudo pacman -S libglvnd glib2`

### RTSP/network stream does not open

Check that the stream URL opens in VLC first. On Linux/macOS, installing `ffmpeg` usually improves stream/video-file compatibility.

## Windows Tkinter/Tcl error

If Windows shows:

```text
_tkinter.TclError: Can't find a usable init.tcl
```

run:

```powershell
powershell -ExecutionPolicy Bypass -File .\fix_tcl_tk_windows.ps1
```

Then close and reopen PowerShell and start the app again with:

```powershell
.\start_station_windows.ps1
```
