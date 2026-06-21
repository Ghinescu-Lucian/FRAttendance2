# Starts the FaceAttendance desktop station on Windows.
# If .venv does not exist yet, run install_windows.ps1 first.

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

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (!(Test-Path $VenvPython)) {
    Write-Host "The virtual environment was not found. Running installer first..." -ForegroundColor Yellow
    powershell -ExecutionPolicy Bypass -File .\install_windows.ps1
}

Set-TclTkEnvironment $VenvPython
& $VenvPython .\desktop_station_app.py
