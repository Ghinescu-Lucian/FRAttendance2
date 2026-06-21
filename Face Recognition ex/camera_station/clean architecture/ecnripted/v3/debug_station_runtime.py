import json
import os
import sys
from pathlib import Path

import moodle_yunet_sface_station as station

print("Python:", sys.executable)
print("CWD:", Path.cwd())
print("SCRIPT_DIR:", Path(__file__).resolve().parent)

for name in ("station_config.json", "station.json", "config.json"):
    for base in (Path.cwd(), Path(__file__).resolve().parent):
        p = base / name
        print("check", p, "exists=", p.exists())
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                print("  api_secret_present=", bool(str(data.get("api_secret", "")).strip()), "length=", len(str(data.get("api_secret", ""))))
                print("  station_private_key_path=", data.get("station_private_key_path"))
            except Exception as exc:
                print("  JSON ERROR:", repr(exc))

print("Before apply: secret_present=", bool(station.MOODLE_API_SECRET and station.MOODLE_API_SECRET not in ("change-this-secret", "CHANGE_ME_USE_LONG_RANDOM_SECRET_FROM_MOODLE")))
station.apply_runtime_options()
print("After apply: secret_present=", bool(station.MOODLE_API_SECRET and station.MOODLE_API_SECRET not in ("change-this-secret", "CHANGE_ME_USE_LONG_RANDOM_SECRET_FROM_MOODLE")), "length=", len(station.MOODLE_API_SECRET or ""))
print("Moodle URL:", station.MOODLE_BASE_URL)
print("CMID:", station.MOODLE_CMID)
print("Station ID:", station.MOODLE_STATION_ID)
print("Verify TLS:", station.MOODLE_VERIFY_TLS)
