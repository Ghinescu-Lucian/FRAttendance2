import dotenv from "dotenv";
dotenv.config();

import path from "node:path";

export const PORT = Number(process.env.PORT || 3000);
export const PROJECT_ROOT = process.cwd();
export const PUBLIC_DIR = path.join(PROJECT_ROOT, "public");
export const UPLOAD_DIR = process.env.UPLOAD_DIR || String.raw`C:\Users\Lucian\Desktop\Dis\images`;
export const HTTPS_KEY_FILE = process.env.HTTPS_KEY_FILE;
export const HTTPS_CERT_FILE = process.env.HTTPS_CERT_FILE;
export const SERVER_BUILD = "modular-sface-upload-speech-2026-05-10";
export const SFACE_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx";
export const SFACE_MODEL_PUBLIC_PATH = path.join(PUBLIC_DIR, "models", "face_recognition_sface_2021dec.onnx");
export const MIN_VALID_SFACE_MODEL_BYTES = 1_000_000;
