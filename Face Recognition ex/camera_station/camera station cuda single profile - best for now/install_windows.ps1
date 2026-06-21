# FaceAttendance Station installer for Windows
# Run from PowerShell:
#   powershell -ExecutionPolicy Bypass -File .\install_windows.ps1

param(
    [switch]$SkipGpuCheck
)

$ErrorActionPreference = "Stop"

function Set-TclTkEnvironment($PythonExe) {
    try {
        $BasePrefix = & $PythonExe -c "import sys; print(sys.base_prefix)"
        if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($BasePrefix)) { return }
        $BasePrefix = $BasePrefix.Trim()
        $Tcl = Join-Path $BasePrefix "tcl\tcl8.6"
        $Tk = Join-Path $BasePrefix "tcl\tk8.6"
        if (Test-Path (Join-Path $Tcl "init.tcl")) {
            $env:TCL_LIBRARY = $Tcl
            Write-Host "TCL_LIBRARY=$Tcl" -ForegroundColor DarkGray
        }
        if (Test-Path (Join-Path $Tk "tk.tcl")) {
            $env:TK_LIBRARY = $Tk
            Write-Host "TK_LIBRARY=$Tk" -ForegroundColor DarkGray
        }
    } catch {
        Write-Host "Could not auto-configure Tcl/Tk paths: $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

function Write-Step($Message) {
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Resolve-PythonCommand {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return @{ Cmd = "py"; Args = @("-3") }
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return @{ Cmd = "python"; Args = @() }
    }
    throw "Python 3 was not found. Install Python 3.10+ from https://www.python.org/downloads/windows/ and make sure 'Add python.exe to PATH' is enabled."
}

Write-Step "Checking Python"
$Python = Resolve-PythonCommand
& $Python.Cmd @($Python.Args) -c "import sys; print(sys.version); assert sys.version_info >= (3, 10), 'Python 3.10 or newer is required'"

Write-Step "Creating virtual environment in .venv"
& $Python.Cmd @($Python.Args) -m venv .venv

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (!(Test-Path $VenvPython)) {
    throw "Virtual environment was not created correctly: $VenvPython not found."
}

Write-Step "Upgrading pip, setuptools, and wheel"
& $VenvPython -m pip install --upgrade pip setuptools wheel

Write-Step "Installing FaceAttendance desktop dependencies"
& $VenvPython -m pip install -r requirements_desktop.txt

Write-Step "Creating runtime folders"
$Folders = @("images", "captures", "unknown_review", "reviewed_embeddings", "reports")
foreach ($Folder in $Folders) {
    New-Item -ItemType Directory -Force -Path (Join-Path $Root $Folder) | Out-Null
}

Write-Step "Configuring Tcl/Tk environment for Tkinter"
Set-TclTkEnvironment $VenvPython

Write-Step "Verifying imports and Tkinter window creation"
& $VenvPython -c "import cv2, numpy, PIL, tkinter as tk; root=tk.Tk(); root.withdraw(); root.destroy(); print('OK: cv2', cv2.__version__); print('OK: Tkinter desktop window test')"

if (-not $SkipGpuCheck) {
    Write-Step "Running GPU diagnostic"
    & $VenvPython .\gpu_check.py
}

Write-Step "Installation complete"
Write-Host "Start the desktop station with:" -ForegroundColor Green
Write-Host "  .\start_station_windows.ps1"
Write-Host "or:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\python.exe .\desktop_station_app.py"
Write-Host "`nNote: pip OpenCV is usually CPU-only. If CUDA devices = 0, the app will run on CPU and safely fall back when GPU mode is enabled." -ForegroundColor Yellow
