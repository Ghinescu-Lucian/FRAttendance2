import dotenv from "dotenv";
import path from "node:path";

dotenv.config();

export const PORT = Number(process.env.PORT || 3000);
export const PROJECT_ROOT = process.cwd();
export const PUBLIC_DIR = path.join(PROJECT_ROOT, "public");
export const UPLOAD_DIR = process.env.UPLOAD_DIR || path.join(PROJECT_ROOT, "images");
export const HTTPS_KEY_FILE = process.env.HTTPS_KEY_FILE;
export const HTTPS_CERT_FILE = process.env.HTTPS_CERT_FILE;
export const SERVER_BUILD = "react-clean-architecture-2026-05-11";
export const SFACE_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx";
export const SFACE_MODEL_PUBLIC_PATH = path.join(PUBLIC_DIR, "models", "face_recognition_sface_2021dec.onnx");
export const MIN_VALID_SFACE_MODEL_BYTES = 1_000_000;
