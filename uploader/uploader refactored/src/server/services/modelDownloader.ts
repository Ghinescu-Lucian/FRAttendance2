import fs from "node:fs";
import https from "node:https";
import path from "node:path";
import { MIN_VALID_SFACE_MODEL_BYTES, SFACE_MODEL_PUBLIC_PATH, SFACE_MODEL_URL } from "../config/serverConfig";

function downloadFile(url: string, outputPath: string, redirectCount = 0): Promise<void> {
  return new Promise((resolve, reject) => {
    if (redirectCount > 5) return reject(new Error(`Too many redirects while downloading ${url}`));

    const request = https.get(url, (response) => {
      const statusCode = response.statusCode || 0;
      const location = response.headers.location;

      if (statusCode >= 300 && statusCode < 400 && location) {
        response.resume();
        return resolve(downloadFile(location, outputPath, redirectCount + 1));
      }

      if (statusCode !== 200) {
        response.resume();
        return reject(new Error(`Download failed for ${url}. HTTP ${statusCode}`));
      }

      const tmpPath = `${outputPath}.part`;
      const fileStream = fs.createWriteStream(tmpPath);
      response.pipe(fileStream);

      fileStream.on("finish", () => fileStream.close(() => {
        fs.renameSync(tmpPath, outputPath);
        resolve();
      }));
      fileStream.on("error", (error) => {
        fs.rmSync(tmpPath, { force: true });
        reject(error);
      });
    });

    request.on("error", reject);
  });
}

export async function ensureSFaceModelFile(): Promise<void> {
  fs.mkdirSync(path.dirname(SFACE_MODEL_PUBLIC_PATH), { recursive: true });

  if (fs.existsSync(SFACE_MODEL_PUBLIC_PATH)) {
    const size = fs.statSync(SFACE_MODEL_PUBLIC_PATH).size;
    if (size >= MIN_VALID_SFACE_MODEL_BYTES) return;
    console.warn(`Existing SFace model file is too small (${size} bytes). Re-downloading.`);
    fs.rmSync(SFACE_MODEL_PUBLIC_PATH, { force: true });
  }

  console.log("SFace ONNX model missing. Downloading it to public/models...");
  await downloadFile(SFACE_MODEL_URL, SFACE_MODEL_PUBLIC_PATH);

  const size = fs.statSync(SFACE_MODEL_PUBLIC_PATH).size;
  if (size < MIN_VALID_SFACE_MODEL_BYTES) {
    fs.rmSync(SFACE_MODEL_PUBLIC_PATH, { force: true });
    throw new Error(`Downloaded SFace model is too small (${size} bytes).`);
  }
  console.log(`SFace ONNX model ready: ${SFACE_MODEL_PUBLIC_PATH} (${size} bytes)`);
}
