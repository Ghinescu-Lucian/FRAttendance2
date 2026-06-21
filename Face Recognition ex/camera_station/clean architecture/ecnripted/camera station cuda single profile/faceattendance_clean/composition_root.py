"""Composition root for the desktop station.

The composition root is the only place that wires presentation to the legacy
station module.  Keeping this boundary explicit is part of the clean-architecture
migration: domain/application modules stay independent from Tkinter/OpenCV.
"""
from __future__ import annotations

from importlib import import_module


def launch_desktop_station() -> None:
    desktop_module = import_module("desktop_station_app")
    app = desktop_module.DesktopStationApp()
    app.mainloop()
