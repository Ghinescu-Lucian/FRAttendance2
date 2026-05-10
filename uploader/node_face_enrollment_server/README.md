# Node Face Enrollment Server

This local Node.js app:

- exposes an API for uploading face images
- saves them directly into:

```text
C:\Users\Ghinescu Lucian\Desktop\Master\DISERTATIE\Face Recognition ex\images
```

- serves a web page that opens the device camera
- draws a green face contour/rectangle
- captures and uploads several JPEG images

## Install

Open PowerShell in this folder:

```powershell
npm install
```

## Run

```powershell
npm start
```

Open on the same PC:

```text
http://localhost:3000
```

## API

Health check:

```text
GET /api/health
```

Upload one image:

```text
POST /api/upload-face
multipart/form-data:
  studentId
  name
  frameIndex
  image
```

Complete enrollment summary:

```text
POST /api/complete-enrollment
JSON:
  studentId
  name
```

## Phone on local Wi-Fi

For a phone browser, camera access usually requires HTTPS unless using `localhost`.

For quick testing from your phone, use ngrok:

```powershell
npm start
ngrok http 3000
```

Then open the HTTPS ngrok URL on your phone.

If you want strict local Wi-Fi without ngrok, use HTTPS with a local certificate later.

## Important

The green contour is a guide only. This version does not verify that the face is really inside the contour. Later, you can add MediaPipe face detection in the browser for validation.
