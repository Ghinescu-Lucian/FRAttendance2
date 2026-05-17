import dotenv from "dotenv";
import fs from "node:fs";
import http from "node:http";
import https from "node:https";
import { createApp } from "./server/app";
import {
  HTTPS_CERT_FILE,
  HTTPS_KEY_FILE,
  PORT,
  SERVER_BUILD,
  UPLOAD_DIR,
} from "./server/config/serverConfig";
import { ensureSFaceModelFile } from "./server/services/modelDownloader";
import { resolveProjectPath } from "./server/utils/files";

dotenv.config();
fs.mkdirSync(UPLOAD_DIR, { recursive: true });

async function startServer(): Promise<void> {
  await ensureSFaceModelFile();
  const app = createApp();

  if (HTTPS_KEY_FILE && HTTPS_CERT_FILE) {
    const keyPath = resolveProjectPath(HTTPS_KEY_FILE);
    const certPath = resolveProjectPath(HTTPS_CERT_FILE);
    if (!fs.existsSync(keyPath))
      throw new Error(`HTTPS key file not found: ${keyPath}`);
    if (!fs.existsSync(certPath))
      throw new Error(`HTTPS cert file not found: ${certPath}`);

    //   https.createServer({ key: fs.readFileSync(keyPath), cert: fs.readFileSync(certPath) }, app).listen(PORT, "0.0.0.0", () => {
    //     console.log(`HTTPS server running: https://localhost:${PORT}`);
    //     console.log(`Phone URL: https://YOUR_PC_WIFI_IP:${PORT}`);
    //     console.log(`Upload directory: ${UPLOAD_DIR}`);
    //     console.log(`Server build: ${SERVER_BUILD}`);
    //   });
    //   return;
    // }

    // http.createServer(app).listen(PORT, "0.0.0.0", () => {
    //   console.log(`HTTP server running: http://localhost:${PORT}`);
    //   console.log("Phone camera will probably NOT work over plain HTTP LAN IP.");
    //   console.log(`Upload directory: ${UPLOAD_DIR}`);
    //   console.log(`Server build: ${SERVER_BUILD}`);
    // });

    https
      .createServer(
        {
          key: fs.readFileSync(keyPath),
          cert: fs.readFileSync(certPath),
        },
        app,
      )
      .listen(PORT, "0.0.0.0", () => {
        console.log(`HTTPS server running: https://localhost:${PORT}`);
        console.log(`Phone URL: https://YOUR_PC_WIFI_IP:${PORT}`);
        console.log(`Upload directory: ${UPLOAD_DIR}`);
        console.log(`Server build: ${SERVER_BUILD}`);
      });

    return;
  }

  http.createServer(app).listen(PORT, "0.0.0.0", () => {
    console.log(`HTTP server running: http://localhost:${PORT}`);
    console.log("Phone camera will probably NOT work over plain HTTP LAN IP.");
    console.log(`Upload directory: ${UPLOAD_DIR}`);
    console.log(`Server build: ${SERVER_BUILD}`);
  });
}

void startServer().catch((error) => {
  console.error(error);
  process.exit(1);
});
