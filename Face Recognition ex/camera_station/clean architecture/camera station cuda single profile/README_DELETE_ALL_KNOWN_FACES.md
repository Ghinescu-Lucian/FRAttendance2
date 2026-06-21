# Delete ALL known faces

This build adds a destructive cleanup button in:

`Reports & Unknown Review -> Known persons -> Delete ALL known faces`

The button is intended for the case where you want the station to forget every known/enrolled person and treat new detections as unknown again.

## What it deletes / clears

After confirmation, the action:

1. Creates an automatic backup ZIP under:

   `deleted_known_faces_backups/known_faces_backup_YYYYMMDD_HHMMSS.zip`

2. Deletes known-face embedding JSON files from:

   - the currently selected **Embeddings source**;
   - local project folders whose names contain `embedding`, for example `embedding_pool_30 + known` or `embedding_pool_lfw_200x5`;
   - `reviewed_embeddings/`, which contains labels created from Unknown review.

3. Resets all assigned labels in `unknown_review/unknown_registry.json`.

   The saved unknown face pictures are not deleted. They remain available in Unknown review and can be labeled again later.

4. Clears the running worker's in-memory known database and known-person report.

5. Temporarily disables known-face reload in the running session, so Moodle/refresh logic does not immediately bring the same known persons back while the station is still running.

## Moodle note

If **Load roster/embeddings from Moodle API** is enabled, this local button cannot delete embeddings stored on the Moodle server. It clears local/reviewed embeddings and the active in-memory worker. Server-side known faces can return after a restart unless they are also removed from Moodle.

## How to add known people again

You can add known people again by:

- selecting/importing a new embeddings source;
- importing a review package;
- assigning a new label to an unknown face in Unknown review.

Assigning or importing reviewed embeddings automatically re-enables known-face recognition.
