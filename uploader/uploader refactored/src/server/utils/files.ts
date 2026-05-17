import fs from "node:fs";
import path from "node:path";
import { PROJECT_ROOT, UPLOAD_DIR } from "../config/serverConfig";

export function safeName(value: unknown, fallback = "unknown"): string {
  const text = String(value || "").trim();
  const cleaned = text.replace(/[^\p{L}\p{N}_.-]+/gu, "_").replace(/^_+|_+$/g, "").slice(0, 80);
  return cleaned || fallback;
}

export function getExtension(originalName: string | undefined, fallback = ".jpg"): string {
  const ext = path.extname(originalName || "").toLowerCase();
  if ([".jpg", ".jpeg", ".png", ".webp", ".bmp", ".json"].includes(ext)) return ext;
  return fallback;
}

export function nextFileIndex(prefix: string, extensionRegex: RegExp): number {
  const files = fs.readdirSync(UPLOAD_DIR);
  let maxIndex = 0;
  for (const file of files) {
    if (!file.startsWith(prefix) || !extensionRegex.test(file)) continue;
    const match = file.match(/_(\d+)\.[^.]+$/);
    const value = match ? Number(match[1]) : NaN;
    if (Number.isFinite(value) && value > maxIndex) maxIndex = value;
  }
  return maxIndex + 1;
}

export function resolveProjectPath(filePath: string): string {
  return path.isAbsolute(filePath) ? filePath : path.join(PROJECT_ROOT, filePath);
}

export function readJsonFile(filePath: string): any {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}
