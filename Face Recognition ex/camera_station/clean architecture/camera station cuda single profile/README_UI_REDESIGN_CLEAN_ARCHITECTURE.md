# UI redesign + clean architecture migration

This build keeps the working recognition pipeline intact, but introduces a cleaner structure around it so future changes do not need to be added only to one very large Tkinter file.

## What changed visually

- New darker modern theme with consistent colors, cards, tabs, buttons, comboboxes, treeviews and status chips.
- Cleaner top header with grouped runtime controls.
- Start/Stop buttons now have distinct accent/danger styling.
- Video controls are grouped in the header and are visually separated from recognition settings.
- The camera preview area has a dedicated video style.
- Runtime counters are now shown as dashboard cards:
  - Faces
  - Known
  - Unknown
  - FPS
  - Video state
- Reports and Unknown Review were restyled with cards and clearer destructive/action buttons.
- Known/unknown tables have modern treeview styling and alternating row tags.

## Clean architecture structure added

The new package is:

```text
faceattendance_clean/
  domain/
    models.py
  application/
    contracts.py
    use_cases.py
  infrastructure/
    legacy_station_adapter.py
  presentation/
    theme.py
    widgets.py
  composition_root.py
```

Layer responsibilities:

- `domain/` contains pure station concepts: media source, playback state, recognition counters, known/unknown reports.
- `application/` contains UI-independent use cases and ports, for example the `VideoPlaybackController` and repository/worker protocols.
- `infrastructure/` contains adapters over existing implementation details, currently the adapter around `moodle_yunet_sface_station.py`.
- `presentation/` contains Tkinter-specific theme and reusable widgets.
- `composition_root.py` wires the desktop app together.

The old files are still present:

```text
desktop_station_app.py
desktop_station_app_zoomfix.py
moodle_yunet_sface_station.py
```

They are still the safest entry points because they contain the already-tested recognition pipeline. The migration is intentionally incremental: the heavy OpenCV/Moodle logic remains stable, while new UI/design/control code is moved into isolated modules.

## Recommended start command

The original command still works:

```powershell
py .\desktop_station_app.py
```

You can also start through the clean composition root:

```powershell
py .\run_station_clean.py
```

## Why this structure helps the dissertation project

This gives you a clearer architectural story:

- Presentation layer: Tkinter desktop UI.
- Application layer: station commands, playback control, ports/protocols.
- Domain layer: recognition/reporting concepts independent from frameworks.
- Infrastructure layer: OpenCV, ONNX Runtime, Moodle, file-system adapters.

This is easier to explain in the dissertation than a single monolithic script, while preserving the working FR behavior.
