# Windows fix: `TclError: Can't find a usable init.tcl`

This error means Python can import `tkinter`, but it cannot find the Tcl/Tk runtime files needed to open the desktop window.

## Fast fix

From the `opencv_station` folder, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\fix_tcl_tk_windows.ps1
```

Then close and reopen PowerShell and start the station:

```powershell
.\start_station_windows.ps1
```

## Manual fix

Check these files exist:

```text
C:\Users\<you>\AppData\Local\Programs\Python\Python313\tcl\tcl8.6\init.tcl
C:\Users\<you>\AppData\Local\Programs\Python\Python313\tcl\tk8.6\tk.tcl
```

If they exist, run:

```powershell
setx TCL_LIBRARY "C:\Users\<you>\AppData\Local\Programs\Python\Python313\tcl\tcl8.6"
setx TK_LIBRARY  "C:\Users\<you>\AppData\Local\Programs\Python\Python313\tcl\tk8.6"
```

Close and reopen PowerShell.

If they do **not** exist, rerun the Python installer, choose **Modify**, and enable **tcl/tk and IDLE**. Installing Python 3.12 from python.org with Tcl/Tk enabled is also a stable option.
