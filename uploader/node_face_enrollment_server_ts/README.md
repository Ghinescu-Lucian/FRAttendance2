# TypeScript Node Face Enrollment Server

This local TypeScript Node.js app:

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

## Development run

```powershell
npm run dev
```

Then open:

```text
http://localhost:3000
```

## Production build/run

```powershell
npm run build
npm start
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
npm run dev
ngrok http 3000
```

Then open the HTTPS ngrok URL on your phone.

## Change upload path

Create `.env` from `.env.example` and edit:

```text
UPLOAD_DIR=C:\Users\Ghinescu Lucian\Desktop\Master\DISERTATIE\Face Recognition ex\images
```
