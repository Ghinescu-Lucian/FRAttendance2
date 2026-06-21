import json
from pathlib import Path

for name in ("station_config.json", "station.json", "config.json"):
    p = Path(name)
    if p.exists():
        print(f"Found: {p.resolve()}")
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"JSON ERROR in {name}: {exc}")
            raise
        print("moodle_base_url =", data.get("moodle_base_url"))
        print("cmid =", data.get("cmid"))
        secret = str(data.get("api_secret", ""))
        print("api_secret loaded =", bool(secret.strip()), f"length={len(secret.strip())}")
        print("station_private_key_path =", data.get("station_private_key_path"))
        break
else:
    print("No station_config.json/station.json/config.json found in current folder")
