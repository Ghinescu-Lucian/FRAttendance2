import multer from "multer";
import { Request } from "express";
import { UPLOAD_DIR } from "../config/serverConfig";
import { getExtension, nextFileIndex, safeName } from "../utils/files";

const imageStorage = multer.diskStorage({
  destination: (_req, _file, cb) => cb(null, UPLOAD_DIR),
  filename: (req: Request, file: Express.Multer.File, cb) => {
    const personName = safeName(req.body.name, "person");
    const ext = getExtension(file.originalname, ".jpg");
    const nextIndex = nextFileIndex(`${personName}_`, /\.(jpg|jpeg|png|webp|bmp)$/i);
    cb(null, `${personName}_${nextIndex}${ext}`);
  },
});

const embeddingStorage = multer.diskStorage({
  destination: (_req, _file, cb) => cb(null, UPLOAD_DIR),
  filename: (req: Request, _file: Express.Multer.File, cb) => {
    const personName = safeName(req.body.name, "person");
    const nextIndex = nextFileIndex(`${personName}_sface_embeddings_`, /\.json$/i);
    cb(null, `${personName}_sface_embeddings_${nextIndex}.json`);
  },
});

export const uploadImage = multer({
  storage: imageStorage,
  limits: { fileSize: 8 * 1024 * 1024 },
  fileFilter: (_req, file, cb) => {
    if (!file.mimetype || !file.mimetype.startsWith("image/")) return cb(new Error("Only image uploads are allowed."));
    cb(null, true);
  },
});

export const uploadEmbeddingFile = multer({
  storage: embeddingStorage,
  limits: { fileSize: 20 * 1024 * 1024 },
  fileFilter: (_req, file, cb) => {
    const isJson = file.mimetype === "application/json" ||
      file.mimetype === "application/octet-stream" ||
      file.originalname.toLowerCase().endsWith(".json");
    if (!isJson) return cb(new Error("Only JSON embedding files are allowed."));
    cb(null, true);
  },
});
