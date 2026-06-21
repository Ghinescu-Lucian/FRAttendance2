# Fixing stubborn default Moodle URL

This version prints the exact config file it loads. Run it like this:

```powershell
py .\main_yunet_sface_many_faces_unknown_fast_short_moodle.py --config "C:\full\path\to\station_config.json"
```

At startup you must see:

```text
[CONFIG] Loaded station config: ...station_config.json
Moodle URL: https://192.168.0.154
Verify TLS: False
```

If you do not see `[CONFIG] Loaded station config`, the script is not reading your JSON.
