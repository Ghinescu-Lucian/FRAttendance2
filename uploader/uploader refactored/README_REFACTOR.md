# React + Clean Architecture refactor

The enrollment page is now rendered by React TypeScript.

## Frontend structure

- `src/public/domain`: core enrollment types and capture steps.
- `src/public/application`: enrollment use-case orchestration and quality/input rules.
- `src/public/infrastructure`: browser APIs, ONNX/face-api services, and HTTP API client.
- `src/public/presentation`: React page plus DOM view adapter used by the controller.
- `src/public/shared`: small shared utilities and app config.

## Run

```powershell
npm install
npm run dev
```

The command builds the React frontend into `public/` and starts the HTTPS Express server.
