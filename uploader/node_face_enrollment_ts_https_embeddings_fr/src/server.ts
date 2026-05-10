import dotenv from "dotenv";
import express, { NextFunction, Request, Response } from "express";
import fs from "node:fs";
import http from "node:http";
import https from "node:https";
import path from "node:path";
import multer from "multer";

dotenv.config();

const app = express();
const PORT = Number(process.env.PORT || 3000);
const PROJECT_ROOT = process.cwd();
const PUBLIC_DIR = path.join(PROJECT_ROOT, "public");

const UPLOAD_DIR =
  process.env.UPLOAD_DIR ||
  String.raw`C:\Users\Ghinescu Lucian\Desktop\Master\DISERTATIE\Face Recognition ex\images`;

const HTTPS_KEY_FILE = process.env.HTTPS_KEY_FILE;
const HTTPS_CERT_FILE = process.env.HTTPS_CERT_FILE;
const SERVER_BUILD = "strict-sface-upload-overlay-2026-05-07";
const SFACE_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx";
const SFACE_MODEL_PUBLIC_PATH = path.join(PUBLIC_DIR, "models", "face_recognition_sface_2021dec.onnx");
const MIN_VALID_SFACE_MODEL_BYTES = 1_000_000;

fs.mkdirSync(UPLOAD_DIR, { recursive: true });

app.use(express.json({ limit: "20mb" }));
app.use(express.urlencoded({ extended: true }));
app.use(
  express.static(PUBLIC_DIR, {
    setHeaders: (res, filePath) => {
      if (/\.(html|js|css|json)$/i.test(filePath)) {
        res.setHeader("Cache-Control", "no-store, no-cache, must-revalidate, proxy-revalidate");
      }
    },
  }),
);


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

      fileStream.on("finish", () => {
        fileStream.close(() => {
          fs.renameSync(tmpPath, outputPath);
          resolve();
        });
      });

      fileStream.on("error", (error) => {
        fs.rmSync(tmpPath, { force: true });
        reject(error);
      });
    });

    request.on("error", reject);
  });
}

async function ensureSFaceModelFile(): Promise<void> {
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

function safeName(value: unknown, fallback = "unknown"): string {
  const text = String(value || "").trim();
  const cleaned = text
    .replace(/[^\p{L}\p{N}_.-]+/gu, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 80);
  return cleaned || fallback;
}

function getExtension(originalName: string | undefined, fallback = ".jpg"): string {
  const ext = path.extname(originalName || "").toLowerCase();
  if ([".jpg", ".jpeg", ".png", ".webp", ".bmp", ".json"].includes(ext)) return ext;
  return fallback;
}


function descriptorNorm(values: number[]): number {
  let sumSq = 0;
  for (const value of values) sumSq += value * value;
  return Math.sqrt(sumSq);
}

function validateSFaceEmbeddingPayload(payload: any): string[] {
  const errors: string[] = [];

  if (!payload || typeof payload !== "object") {
    return ["Embedding file must contain a JSON object."];
  }

  const model = payload.model || {};
  const family = String(model.family || "").toLowerCase();
  const recognizer = String(model.recognizer || "").toLowerCase();

  if (family !== "opencv") {
    errors.push(`model.family must be 'opencv', got '${family || "missing"}'. This looks like an old face-api file.`);
  }

  if (recognizer !== "sface") {
    errors.push(`model.recognizer must be 'sface', got '${recognizer || "missing"}'. This looks like an old face-api file.`);
  }

  if (Number(model.descriptorLength) !== 128) {
    errors.push(`model.descriptorLength must be 128, got '${model.descriptorLength ?? "missing"}'.`);
  }

  if (!Array.isArray(payload.captures) || payload.captures.length === 0) {
    errors.push("captures must be a non-empty array.");
    return errors;
  }

  for (let i = 0; i < payload.captures.length; i++) {
    const capture = payload.captures[i];
    const descriptor = capture?.descriptor;
    const label = capture?.label || capture?.pose || `capture ${i + 1}`;

    if (!Array.isArray(descriptor)) {
      errors.push(`${label}: descriptor must be an array.`);
      continue;
    }

    if (descriptor.length !== 128) {
      errors.push(`${label}: descriptor must have 128 numbers, got ${descriptor.length}.`);
      continue;
    }

    const numbers = descriptor.map((value: unknown) => Number(value));
    if (!numbers.every((value: number) => Number.isFinite(value))) {
      errors.push(`${label}: descriptor contains non-numeric values.`);
      continue;
    }

    const norm = descriptorNorm(numbers);
    if (Math.abs(norm - 1.0) > 0.08) {
      errors.push(`${label}: descriptor is not L2-normalized enough; norm=${norm.toFixed(4)}, expected about 1.0.`);
    }
  }

  return errors;
}

function readJsonFile(filePath: string): any {
  const text = fs.readFileSync(filePath, "utf8");
  return JSON.parse(text);
}

function nextFileIndex(prefix: string, extensionRegex: RegExp): number {
  const files = fs.readdirSync(UPLOAD_DIR);

  let maxIndex = 0;

  for (const file of files) {
    if (!file.startsWith(prefix)) {
      continue;
    }

    if (!extensionRegex.test(file)) {
      continue;
    }

    const match = file.match(/_(\d+)\.[^.]+$/);

    if (!match) {
      continue;
    }

    const value = Number(match[1]);

    if (Number.isFinite(value) && value > maxIndex) {
      maxIndex = value;
    }
  }

  return maxIndex + 1;
}

const imageStorage = multer.diskStorage({
  destination: (_req: Request, _file: Express.Multer.File, cb) => cb(null, UPLOAD_DIR),
  filename: (req: Request, file: Express.Multer.File, cb) => {
    const personName = safeName(req.body.name, "person");
    const ext = getExtension(file.originalname, ".jpg");

    // Simple image names for your face-recognition images folder:
    // GhinescuLucian_1.jpg, GhinescuLucian_2.jpg, ...
    const nextIndex = nextFileIndex(`${personName}_`, /\.(jpg|jpeg|png|webp|bmp)$/i);

    cb(null, `${personName}_${nextIndex}${ext}`);
  },
});

const embeddingStorage = multer.diskStorage({
  destination: (_req: Request, _file: Express.Multer.File, cb) => cb(null, UPLOAD_DIR),
  filename: (req: Request, _file: Express.Multer.File, cb) => {
    const personName = safeName(req.body.name, "person");

    // Explicit SFace embedding names:
    // GhinescuLucian_sface_embeddings_1.json, GhinescuLucian_sface_embeddings_2.json, ...
    const nextIndex = nextFileIndex(`${personName}_sface_embeddings_`, /\.json$/i);

    cb(null, `${personName}_sface_embeddings_${nextIndex}.json`);
  },
});

const uploadImage = multer({
  storage: imageStorage,
  limits: { fileSize: 8 * 1024 * 1024 },
  fileFilter: (_req, file, cb) => {
    if (!file.mimetype || !file.mimetype.startsWith("image/")) return cb(new Error("Only image uploads are allowed."));
    cb(null, true);
  },
});

const uploadEmbeddingFile = multer({
  storage: embeddingStorage,
  limits: { fileSize: 20 * 1024 * 1024 },
  fileFilter: (_req, file, cb) => {
    const isJson =
      file.mimetype === "application/json" ||
      file.mimetype === "application/octet-stream" ||
      file.originalname.toLowerCase().endsWith(".json");
    if (!isJson) return cb(new Error("Only JSON embedding files are allowed."));
    cb(null, true);
  },
});

app.get("/api/health", (_req: Request, res: Response) => {
  const modelExists = fs.existsSync(SFACE_MODEL_PUBLIC_PATH);
  const modelSize = modelExists ? fs.statSync(SFACE_MODEL_PUBLIC_PATH).size : 0;
  res.json({
    ok: true,
    serverBuild: SERVER_BUILD,
    uploadDir: UPLOAD_DIR,
    publicDir: PUBLIC_DIR,
    https: Boolean(HTTPS_KEY_FILE && HTTPS_CERT_FILE),
    sfaceModel: {
      url: "/models/face_recognition_sface_2021dec.onnx",
      exists: modelExists,
      size: modelSize,
    },
  });
});

app.post("/api/upload-face", (_req: Request, res: Response) => {
  return res.status(410).json({
    ok: false,
    error: "Image uploads are disabled. This enrollment server accepts only OpenCV SFace embedding JSON files.",
  });
});

app.post("/api/upload-embedding-file", uploadEmbeddingFile.single("embeddingFile"), (req: Request, res: Response) => {
  if (!req.file) return res.status(400).json({ ok: false, error: "Missing embedding file. Expected form field: embeddingFile" });

  try {
    const payload = readJsonFile(req.file.path);
    const validationErrors = validateSFaceEmbeddingPayload(payload);

    if (validationErrors.length > 0) {
      fs.unlinkSync(req.file.path);
      return res.status(400).json({
        ok: false,
        error: "Rejected embedding file because it is not an OpenCV SFace embedding file.",
        details: validationErrors,
      });
    }

    return res.json({
      ok: true,
      type: "sface-embedding-file",
      file: { filename: req.file.filename, path: req.file.path, size: req.file.size, mimetype: req.file.mimetype },
      metadata: {
        studentId: typeof req.body.studentId === "string" ? req.body.studentId : null,
        name: typeof req.body.name === "string" ? req.body.name : null,
        model: typeof req.body.model === "string" ? req.body.model : null,
        clientBuild: typeof req.body.clientBuild === "string" ? req.body.clientBuild : null,
        captures: payload.captures.length,
        recognizer: payload.model.recognizer,
      },
    });
  } catch (error) {
    if (req.file?.path && fs.existsSync(req.file.path)) fs.unlinkSync(req.file.path);
    const message = error instanceof Error ? error.message : "Invalid JSON embedding file.";
    return res.status(400).json({ ok: false, error: message });
  }
});

app.post("/api/complete-enrollment", (req: Request, res: Response) => {
  const personName = safeName(req.body.name, "person");
  const studentId = safeName(req.body.studentId, "student");
  const files = fs.readdirSync(UPLOAD_DIR).filter((file) => file.startsWith(`${personName}_`));
  const imageFiles = files.filter((file) => /^.+_\d+\.(jpg|jpeg|png|webp|bmp)$/i.test(file));
  const embeddingFiles = files.filter((file) => file.startsWith(`${personName}_sface_embeddings_`) && file.endsWith(".json"));
  res.json({ ok: true, name: req.body.name, studentId: req.body.studentId, savedImages: imageFiles.length, savedEmbeddingFiles: embeddingFiles.length, uploadDir: UPLOAD_DIR, files });
});

app.use((err: Error, _req: Request, res: Response, _next: NextFunction) => {
  console.error(err);
  res.status(500).json({ ok: false, error: err.message || "Server error" });
});

function resolveProjectPath(filePath: string): string {
  return path.isAbsolute(filePath) ? filePath : path.join(PROJECT_ROOT, filePath);
}

async function startServer(): Promise<void> {
  await ensureSFaceModelFile();

  if (HTTPS_KEY_FILE && HTTPS_CERT_FILE) {
    const keyPath = resolveProjectPath(HTTPS_KEY_FILE);
    const certPath = resolveProjectPath(HTTPS_CERT_FILE);
    if (!fs.existsSync(keyPath)) throw new Error(`HTTPS key file not found: ${keyPath}`);
    if (!fs.existsSync(certPath)) throw new Error(`HTTPS cert file not found: ${certPath}`);
    https.createServer({ key: fs.readFileSync(keyPath), cert: fs.readFileSync(certPath) }, app).listen(PORT, "0.0.0.0", () => {
      console.log(`HTTPS server running: https://localhost:${PORT}`);
      console.log(`Phone URL: https://YOUR_PC_WIFI_IP:${PORT}`);
      console.log(`Upload directory: ${UPLOAD_DIR}`);
      console.log(`Server build: ${SERVER_BUILD}`);
    });
  } else {
    http.createServer(app).listen(PORT, "0.0.0.0", () => {
      console.log(`HTTP server running: http://localhost:${PORT}`);
      console.log("Phone camera will probably NOT work over plain HTTP LAN IP.");
      console.log(`Upload directory: ${UPLOAD_DIR}`);
      console.log(`Server build: ${SERVER_BUILD}`);
    });
  }
}

void startServer().catch((error) => {
  console.error(error);
  process.exit(1);
});
