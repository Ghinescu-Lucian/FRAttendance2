# Fixes the common Windows Python/Tkinter error:
#   TclError: Can't find a usable init.tcl
# Run from PowerShell inside the opencv_station folder:
#   powershell -ExecutionPolicy Bypass -File .\fix_tcl_tk_windows.ps1

$ErrorActionPreference = "Stop"

function Resolve-PythonExe {
    $Root = Split-Path -Parent $MyInvocation.MyCommand.Path
    $VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
    if (Test-Path $VenvPython) { return $VenvPython }
    if (Get-Command py -ErrorAction SilentlyContinue) { return "py" }
    if (Get-Command python -ErrorAction SilentlyContinue) { return "python" }
    throw "Python was not found."
}

$PythonExe = Resolve-PythonExe
$BasePrefix = & $PythonExe -c "import sys; print(sys.base_prefix)"
$BasePrefix = $BasePrefix.Trim()

$Tcl = Join-Path $BasePrefix "tcl\tcl8.6"
$Tk = Join-Path $BasePrefix "tcl\tk8.6"

Write-Host "Python base: $BasePrefix" -ForegroundColor Cyan
Write-Host "Checking: $Tcl\init.tcl"
Write-Host "Checking: $Tk\tk.tcl"

if (!(Test-Path (Join-Path $Tcl "init.tcl"))) {
    Write-Host "Missing init.tcl. Rerun the Python installer, choose Modify, and enable 'tcl/tk and IDLE'." -ForegroundColor Red
    Write-Host "Alternative: install Python 3.12 from python.org and run install_windows.ps1 again." -ForegroundColor Yellow
    exit 1
}
if (!(Test-Path (Join-Path $Tk "tk.tcl"))) {
    Write-Host "Missing tk.tcl. Rerun the Python installer, choose Modify, and enable 'tcl/tk and IDLE'." -ForegroundColor Red
    Write-Host "Alternative: install Python 3.12 from python.org and run install_windows.ps1 again." -ForegroundColor Yellow
    exit 1
}

$env:TCL_LIBRARY = $Tcl
$env:TK_LIBRARY = $Tk

Write-Host "Testing Tkinter in this terminal..." -ForegroundColor Cyan
& $PythonExe -c "import tkinter as tk; root=tk.Tk(); root.withdraw(); root.destroy(); print('OK: Tkinter works')"

Write-Host "Saving TCL_LIBRARY and TK_LIBRARY for future terminals..." -ForegroundColor Cyan
setx TCL_LIBRARY "$Tcl" | Out-Null
setx TK_LIBRARY "$Tk" | Out-Null

Write-Host "Done. Close and reopen PowerShell, then run:" -ForegroundColor Green
Write-Host "  .\start_station_windows.ps1"
