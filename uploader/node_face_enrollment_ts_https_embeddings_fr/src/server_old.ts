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

fs.mkdirSync(UPLOAD_DIR, { recursive: true });

app.use(express.json({ limit: "20mb" }));
app.use(express.urlencoded({ extended: true }));
app.use(express.static(PUBLIC_DIR));

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

    // Simple embedding names:
    // GhinescuLucian_embeddings_1.json, GhinescuLucian_embeddings_2.json, ...
    const nextIndex = nextFileIndex(`${personName}_embeddings_`, /\.json$/i);

    cb(null, `${personName}_embeddings_${nextIndex}.json`);
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
  res.json({ ok: true, uploadDir: UPLOAD_DIR, https: Boolean(HTTPS_KEY_FILE && HTTPS_CERT_FILE) });
});

app.post("/api/upload-face", uploadImage.single("image"), (req: Request, res: Response) => {
  if (!req.file) return res.status(400).json({ ok: false, error: "Missing image file. Expected form field: image" });
  return res.json({
    ok: true,
    type: "image",
    file: { filename: req.file.filename, path: req.file.path, size: req.file.size, mimetype: req.file.mimetype },
    metadata: {
      studentId: typeof req.body.studentId === "string" ? req.body.studentId : null,
      name: typeof req.body.name === "string" ? req.body.name : null,
      frameIndex: typeof req.body.frameIndex === "string" ? req.body.frameIndex : null,
      pose: typeof req.body.pose === "string" ? req.body.pose : null,
    },
  });
});

app.post("/api/upload-embedding-file", uploadEmbeddingFile.single("embeddingFile"), (req: Request, res: Response) => {
  if (!req.file) return res.status(400).json({ ok: false, error: "Missing embedding file. Expected form field: embeddingFile" });
  return res.json({
    ok: true,
    type: "embedding-file",
    file: { filename: req.file.filename, path: req.file.path, size: req.file.size, mimetype: req.file.mimetype },
    metadata: {
      studentId: typeof req.body.studentId === "string" ? req.body.studentId : null,
      name: typeof req.body.name === "string" ? req.body.name : null,
      model: typeof req.body.model === "string" ? req.body.model : null,
    },
  });
});

app.post("/api/complete-enrollment", (req: Request, res: Response) => {
  const personName = safeName(req.body.name, "person");
  const studentId = safeName(req.body.studentId, "student");
  const files = fs.readdirSync(UPLOAD_DIR).filter((file) => file.startsWith(`${personName}_`));
  const imageFiles = files.filter((file) => /^.+_\d+\.(jpg|jpeg|png|webp|bmp)$/i.test(file));
  const embeddingFiles = files.filter((file) => file.startsWith(`${personName}_embeddings_`) && file.endsWith(".json"));
  res.json({ ok: true, name: req.body.name, studentId: req.body.studentId, savedImages: imageFiles.length, savedEmbeddingFiles: embeddingFiles.length, uploadDir: UPLOAD_DIR, files });
});

app.use((err: Error, _req: Request, res: Response, _next: NextFunction) => {
  console.error(err);
  res.status(500).json({ ok: false, error: err.message || "Server error" });
});

function resolveProjectPath(filePath: string): string {
  return path.isAbsolute(filePath) ? filePath : path.join(PROJECT_ROOT, filePath);
}

if (HTTPS_KEY_FILE && HTTPS_CERT_FILE) {
  const keyPath = resolveProjectPath(HTTPS_KEY_FILE);
  const certPath = resolveProjectPath(HTTPS_CERT_FILE);
  if (!fs.existsSync(keyPath)) throw new Error(`HTTPS key file not found: ${keyPath}`);
  if (!fs.existsSync(certPath)) throw new Error(`HTTPS cert file not found: ${certPath}`);
  https.createServer({ key: fs.readFileSync(keyPath), cert: fs.readFileSync(certPath) }, app).listen(PORT, "0.0.0.0", () => {
    console.log(`HTTPS server running: https://localhost:${PORT}`);
    console.log(`Phone URL: https://YOUR_PC_WIFI_IP:${PORT}`);
    console.log(`Upload directory: ${UPLOAD_DIR}`);
  });
} else {
  http.createServer(app).listen(PORT, "0.0.0.0", () => {
    console.log(`HTTP server running: http://localhost:${PORT}`);
    console.log("Phone camera will probably NOT work over plain HTTP LAN IP.");
    console.log(`Upload directory: ${UPLOAD_DIR}`);
  });
}
