import dotenv from "dotenv";
import express, { NextFunction, Request, Response } from "express";
import fs from "node:fs";
import path from "node:path";
import multer from "multer";

dotenv.config();

const app = express();

const PORT = Number(process.env.PORT || 3000);

const PROJECT_ROOT = process.cwd();
const PUBLIC_DIR = path.join(PROJECT_ROOT, "public");

// Your requested destination folder.
// You can override it with UPLOAD_DIR in .env if needed.
const UPLOAD_DIR =
  process.env.UPLOAD_DIR ||
  String.raw`C:\Users\Ghinescu Lucian\Desktop\Master\DISERTATIE\Face Recognition ex\images`;

fs.mkdirSync(UPLOAD_DIR, { recursive: true });

app.use(express.json());
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

function getExtension(originalName: string | undefined): string {
  const ext = path.extname(originalName || "").toLowerCase();

  if ([".jpg", ".jpeg", ".png", ".webp", ".bmp"].includes(ext)) {
    return ext;
  }

  return ".jpg";
}

const storage = multer.diskStorage({
  destination: (_req: Request, _file: Express.Multer.File, cb) => {
    cb(null, UPLOAD_DIR);
  },

  filename: (req: Request, file: Express.Multer.File, cb) => {
    const personName = safeName(req.body.name, "person");
    const studentId = safeName(req.body.studentId, "student");
    const frameIndex = safeName(req.body.frameIndex, "0");

    const timestamp = new Date()
      .toISOString()
      .replace(/[:.]/g, "-")
      .replace("T", "_")
      .replace("Z", "");

    const ext = getExtension(file.originalname);

    cb(null, `${personName}_${studentId}_${timestamp}_frame_${frameIndex}${ext}`);
  },
});

const upload = multer({
  storage,
  limits: {
    fileSize: 8 * 1024 * 1024,
  },
  fileFilter: (_req, file, cb) => {
    if (!file.mimetype || !file.mimetype.startsWith("image/")) {
      cb(new Error("Only image uploads are allowed."));
      return;
    }

    cb(null, true);
  },
});

interface UploadResponse {
  ok: boolean;
  file: {
    filename: string;
    path: string;
    size: number;
    mimetype: string;
  };
  metadata: {
    studentId: string | null;
    name: string | null;
    frameIndex: string | null;
  };
}

app.get("/api/health", (_req: Request, res: Response) => {
  res.json({
    ok: true,
    uploadDir: UPLOAD_DIR,
  });
});

app.post(
  "/api/upload-face",
  upload.single("image"),
  (req: Request, res: Response<UploadResponse | { ok: false; error: string }>) => {
    if (!req.file) {
      return res.status(400).json({
        ok: false,
        error: "Missing image file. Expected form field: image",
      });
    }

    return res.json({
      ok: true,
      file: {
        filename: req.file.filename,
        path: req.file.path,
        size: req.file.size,
        mimetype: req.file.mimetype,
      },
      metadata: {
        studentId: typeof req.body.studentId === "string" ? req.body.studentId : null,
        name: typeof req.body.name === "string" ? req.body.name : null,
        frameIndex: typeof req.body.frameIndex === "string" ? req.body.frameIndex : null,
      },
    });
  }
);

app.post("/api/complete-enrollment", (req: Request, res: Response) => {
  const personName = safeName(req.body.name, "person");
  const studentId = safeName(req.body.studentId, "student");

  const files = fs
    .readdirSync(UPLOAD_DIR)
    .filter((file) => file.startsWith(`${personName}_${studentId}_`));

  res.json({
    ok: true,
    name: req.body.name,
    studentId: req.body.studentId,
    savedImages: files.length,
    uploadDir: UPLOAD_DIR,
    files,
  });
});

app.use((err: Error, _req: Request, res: Response, _next: NextFunction) => {
  console.error(err);

  res.status(500).json({
    ok: false,
    error: err.message || "Server error",
  });
});

app.listen(PORT, "0.0.0.0", () => {
  console.log(`Server running: http://localhost:${PORT}`);
  console.log(`Upload directory: ${UPLOAD_DIR}`);
});
