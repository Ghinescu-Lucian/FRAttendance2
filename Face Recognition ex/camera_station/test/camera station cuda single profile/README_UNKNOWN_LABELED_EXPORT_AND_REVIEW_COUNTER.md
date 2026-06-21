# Unknown review export/import update

This build changes the **Export unknown + labeled faces** button in the Reports / Unknown review tab.

## Export content

The exported ZIP now contains the full review database:

- `unknown_review/unknown_registry.json` - all review records, including unlabeled and already labeled tracks.
- `unknown_review/images/` - all captured face images from the review database.
- `reviewed_embeddings/` - embeddings created when an unknown face is assigned to a label.
- `labeled_faces/<label>/` - convenience copies of images that belong to already labeled tracks, grouped by label/person name.

The canonical import data is still `unknown_review/` plus `reviewed_embeddings/`. The `labeled_faces/` folder is added so the ZIP can also be inspected manually or used as a small labeled image dataset.

## Review counter

The Unknown review header now shows a live summary:

- persons detected in the current frame;
- currently detected known persons;
- currently detected unknown persons;
- total review tracks;
- how many review tracks are labeled / unlabeled;
- total captured pictures in the review list.

Example:

```text
Detected now: 2 persons (known 1, unknown 1) | Review list: 12 total, 4 labeled, 8 unlabeled | Pictures: 37
```
