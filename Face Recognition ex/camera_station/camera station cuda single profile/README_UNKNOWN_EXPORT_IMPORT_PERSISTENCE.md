# Unknown faces export/import and persistence fix

This version adds export/import support for the `Unknown review` database and fixes the case where a face assigned from `Unknown review` stays visible in the list but is not recognized in later sessions or after a Moodle refresh.

## New buttons

In **Reports / Unknown review** there are two new buttons:

- **Export unknown faces**
- **Import unknown faces**

The exported ZIP package contains:

- `unknown_review/unknown_registry.json`
- all saved unknown face crops from `unknown_review/images/`
- reviewed/labeled embeddings from `reviewed_embeddings/`

Import merges the package with the current local database. Existing records are kept. If an imported `UNK-xxxx` id already exists locally, the imported record receives a new id.

## Persistence fix for labeled unknown faces

When you assign an unknown face to a name, the app writes a permanent SFace embedding file into:

```text
reviewed_embeddings/
```

Before this patch, those reviewed embeddings were saved but were not always loaded back into the active recognition database. That made the assigned person appear in the unknown list but not be recognized after restarting the app. In Moodle mode, a roster refresh could also replace the local known-feature database and remove the reviewed local labels after a few minutes.

This version fixes both cases:

1. On station start, the app loads the selected/Moodle embeddings and then merges all files from `reviewed_embeddings/`.
2. When Moodle refreshes the known-feature database, the app re-merges `reviewed_embeddings/` immediately.
3. When you assign a label while the station or the video-recognition worker is running, the new embedding is injected into all active workers.
4. When you import a package, reviewed embeddings are also injected into active workers.

## Recommended use

After assigning an unknown as a real person/student:

1. Keep the exact same name format you use in your Moodle/embedding database.
2. Let the app continue running; the active worker is updated immediately.
3. Restarting the app should still recognize the person because the reviewed embedding is now loaded from `reviewed_embeddings/`.

If you want to move unknown review data to another machine, use **Export unknown faces**, then **Import unknown faces** on the other machine.
