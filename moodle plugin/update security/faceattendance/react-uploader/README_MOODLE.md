# React uploader source for `mod_faceattendance`

This folder keeps the editable React/Vite face-embedding recorder inside the Moodle plugin.

Moodle does **not** run this folder directly. Moodle serves the compiled production bundle from:

```text
mod/faceattendance/uploader/
```

The Moodle wrapper page is:

```text
mod/faceattendance/recorder.php
```

`recorder.php` injects this object before loading the React bundle:

```js
window.FACEATTENDANCE_CONTEXT = {
  cmid: 12,
  userid: 25,
  sesskey: "...",
  studentId: "25",
  studentName: "Student Name",
  saveEmbeddingUrl: ".../mod/faceattendance/api/save_embedding_file.php",
  sfaceModelUrl: ".../mod/faceattendance/uploader/models/face_recognition_sface_2021dec.onnx",
  returnUrl: ".../mod/faceattendance/register.php?id=12"
};
```

The React app generates the SFace embedding in the browser and posts the JSON file to Moodle. Raw face images are not uploaded.

## Edit/build workflow

From this directory:

```bash
npm install
npm run build
```

The build output is written directly to:

```text
../uploader/assets/
```

Keep this model file in place because the React app loads it from Moodle:

```text
../uploader/models/face_recognition_sface_2021dec.onnx
```

## Runtime/development split

- `uploader/` is the compiled app Moodle uses at runtime.
- `react-uploader/` is the editable React/Vite source.
- Do not include `node_modules` in the Moodle plugin ZIP.
