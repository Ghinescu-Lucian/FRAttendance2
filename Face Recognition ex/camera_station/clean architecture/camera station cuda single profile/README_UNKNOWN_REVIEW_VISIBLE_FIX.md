# Unknown Review visible controls fix

This patch changes the Reports & Unknown Review tab layout so the Unknown controls are no longer hidden on narrow screens.

Changes:

- The Known persons and Unknown review sections are stacked vertically instead of side-by-side.
- Unknown buttons are placed above the Unknown table:
  - Sort by date
  - Sort by detections
  - Delete selected
  - Delete all unlabeled
- The Known and Unknown tables now have vertical scrollbars.
- The Unknown table can also be sorted by clicking the Detections or Last seen column headers.
- The delete-all action still deletes only unassigned/unlabeled unknown tracks and their saved pictures.

Apply by extracting this archive over your existing opencv_station folder.
