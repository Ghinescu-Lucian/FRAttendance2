# Desktop UI layout fix

This patch fixes the desktop app controls becoming hidden when the right-side settings panel grows too tall.

Changes:

- The `Start` and `Stop` buttons are now fixed in the top header bar.
- The right-side camera/settings panel is now vertically scrollable.
- The old Start/Stop block at the bottom of the settings panel was removed so it cannot be clipped off-screen.

Modified files:

- `desktop_station_app.py`
- `desktop_station_app_zoomfix.py`

Run as usual:

```powershell
py .\desktop_station_app.py
```

If your screen is small, use the mouse wheel over the settings panel to reach the lower options.
