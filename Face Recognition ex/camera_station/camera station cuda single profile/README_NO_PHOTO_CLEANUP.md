# Unknown no-photo cleanup

Adds a third cleanup button in **Reports & Unknown Review**:

- `Delete selected unknowns` - deletes the selected unknown tracks and their saved images.
- `Delete all unlabeled` - deletes all unlabeled unknown tracks, including tracks with images.
- `Delete records without pictures` - deletes only unlabeled unknown tracks that have `0` saved pictures.

The new cleanup keeps labeled/assigned tracks even if they have no pictures, so reviewed records are not removed accidentally.

Files changed:

- `desktop_station_app.py`
- `desktop_station_app_zoomfix.py`
