# Known-face stop limit and Unknown review controls

This patch adds two operational fixes for crowded classroom tests.

## Known persons

A known person is now stopped after a configurable number of accepted recognitions.
The default is:

```json
"known_stop_after_detections": 150
```

After the limit is reached, that person's counter is capped, the box remains hidden,
and future overlapping detections are reused as resolved skip targets instead of
running SFace repeatedly. Use `0` to disable the limit.

Desktop UI control: **Stop known after detections**.

CLI option:

```powershell
py .\main_yunet_sface_many_faces_unknown_fast_short_moodle.py --config .\station_config.json --known-stop-after-detections 150
```

## Unknown review

The Reports & Unknown Review tab now again includes:

- Sort by date
- Sort by detections
- Delete selected
- Delete all unlabeled

Deleting an unknown track also removes its saved preview pictures from
`unknown_review/images`.
