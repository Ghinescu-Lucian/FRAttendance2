# Face Attendance plugin with React uploader

This package keeps the React/Vite uploader application inside the Moodle plugin.

Runtime files used by Moodle:

```text
uploader/                         compiled React assets + SFace model
recorder.php                      Moodle wrapper that loads the compiled React app
api/save_embedding_file.php       Moodle endpoint that stores generated embeddings
```

Editable React source:

```text
react-uploader/
```

To edit the UI:

```bash
cd react-uploader
npm install
npm run build
```

The generated bundle updates `../uploader/assets`, which is what Moodle loads.
