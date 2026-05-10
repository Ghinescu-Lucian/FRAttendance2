const express = require("express");
const multer = require("multer");
const path = require("path");
const fs = require("fs");
require("dotenv").config();

const app = express();

const PORT = Number(process.env.PORT || 3000);

// Your requested destination folder.
// You can override it with UPLOAD_DIR in .env if needed.
const UPLOAD_DIR =
  process.env.UPLOAD_DIR ||
  String.raw`C:\Users\Ghinescu Lucian\Desktop\Master\DISERTATIE\Face Recognition ex\images`;

fs.mkdirSync(UPLOAD_DIR, { recursive: true });

app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(express.static(path.join(__dirname, "public")));

function safeName(value, fallback = "unknown") {
  const text = String(value || "").trim();
  const cleaned = text
    .replace(/[^\p{L}\p{N}_.-]+/gu, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 80);

  return cleaned || fallback;
}

const storage = multer.diskStorage({
  destination: function (_req, _file, cb) {
    cb(null, UPLOAD_DIR);
  },

  filename: function (req, file, cb) {
    const personName = safeName(req.body.name, "person");
    const studentId = safeName(req.body.studentId, "student");
    const frameIndex = safeName(req.body.frameIndex, "0");

    const timestamp = new Date()
      .toISOString()
      .replace(/[:.]/g, "-")
      .replace("T", "_")
      .replace("Z", "");

    const ext = path.extname(file.originalname || "").toLowerCase() || ".jpg";

    cb(null, `${personName}_${studentId}_${timestamp}_frame_${frameIndex}${ext}`);
  },
});

const upload = multer({
  storage,
  limits: {
    fileSize: 8 * 1024 * 1024, // 8 MB
  },
  fileFilter: function (_req, file, cb) {
    if (!file.mimetype || !file.mimetype.startsWith("image/")) {
      return cb(new Error("Only image uploads are allowed."));
    }

    cb(null, true);
  },
});

app.get("/api/health", (_req, res) => {
  res.json({
    ok: true,
    uploadDir: UPLOAD_DIR,
  });
});

app.post("/api/upload-face", upload.single("image"), (req, res) => {
  if (!req.file) {
    return res.status(400).json({
      ok: false,
      error: "Missing image file. Expected form field: image",
    });
  }

  res.json({
    ok: true,
    file: {
      filename: req.file.filename,
      path: req.file.path,
      size: req.file.size,
      mimetype: req.file.mimetype,
    },
    metadata: {
      studentId: req.body.studentId || null,
      name: req.body.name || null,
      frameIndex: req.body.frameIndex || null,
    },
  });
});

app.post("/api/complete-enrollment", (req, res) => {
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

app.use((err, _req, res, _next) => {
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
