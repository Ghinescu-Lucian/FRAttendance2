import fs from "node:fs";
import { Router } from "express";
import { PUBLIC_DIR, SERVER_BUILD, SFACE_MODEL_PUBLIC_PATH, UPLOAD_DIR, HTTPS_CERT_FILE, HTTPS_KEY_FILE } from "../config/serverConfig";
import { uploadEmbeddingFile } from "../services/uploadStorage";
import { readJsonFile, safeName } from "../utils/files";
import { validateSFaceEmbeddingPayload } from "../validators/sfacePayloadValidator";

export const enrollmentRoutes = Router();

enrollmentRoutes.get("/health", (_req, res) => {
  const modelExists = fs.existsSync(SFACE_MODEL_PUBLIC_PATH);
  const modelSize = modelExists ? fs.statSync(SFACE_MODEL_PUBLIC_PATH).size : 0;
  res.json({
    ok: true,
    serverBuild: SERVER_BUILD,
    uploadDir: UPLOAD_DIR,
    publicDir: PUBLIC_DIR,
    https: Boolean(HTTPS_KEY_FILE && HTTPS_CERT_FILE),
    sfaceModel: { url: "/models/face_recognition_sface_2021dec.onnx", exists: modelExists, size: modelSize },
  });
});

enrollmentRoutes.post("/upload-face", (_req, res) => {
  return res.status(410).json({
    ok: false,
    error: "Image uploads are disabled. This enrollment server accepts only OpenCV SFace embedding JSON files.",
  });
});

enrollmentRoutes.post("/upload-embedding-file", uploadEmbeddingFile.single("embeddingFile"), (req, res) => {
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

enrollmentRoutes.post("/complete-enrollment", (req, res) => {
  const personName = safeName(req.body.name, "person");
  const studentId = safeName(req.body.studentId, "student");
  const files = fs.readdirSync(UPLOAD_DIR).filter((file) => file.startsWith(`${personName}_`));
  const imageFiles = files.filter((file) => /^.+_\d+\.(jpg|jpeg|png|webp|bmp)$/i.test(file));
  const embeddingFiles = files.filter((file) => file.startsWith(`${personName}_sface_embeddings_`) && file.endsWith(".json"));
  res.json({ ok: true, name: req.body.name, studentId, savedImages: imageFiles.length, savedEmbeddingFiles: embeddingFiles.length, uploadDir: UPLOAD_DIR, files });
});
