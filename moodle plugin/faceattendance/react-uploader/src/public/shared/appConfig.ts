const moodleContext = window.FACEATTENDANCE_CONTEXT;

export const FACE_API_MODEL_URL = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/";
export const SFACE_MODEL_URL = moodleContext?.sfaceModelUrl || "/models/face_recognition_sface_2021dec.onnx";
export const ORT_WASM_PATH = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
export const ENROLLMENT_APP_BUILD = "moodle-sface-recorder-2026-05-17-unknown-thumbnails-dedupe";
export const UPLOAD_ACCEPTED_IMAGES = false;

export const SFACE_SIZE = 112;
export const SFACE_REFERENCE_POINTS = [
  { x: 38.2946, y: 51.6963 },
  { x: 73.5318, y: 51.5014 },
  { x: 56.0252, y: 71.7366 },
  { x: 41.5493, y: 92.3655 },
  { x: 70.7299, y: 92.2041 },
] as const;

export const QUALITY_RULES = {
  minDetectionScore: 0.5,
  minBlur: 45,
  minBrightness: 45,
  maxBrightness: 225,
  minFaceHeightRatio: 0.18,
  maxFaceHeightRatio: 0.7,
  stableFramesRequired: 3,
  captureTimeoutMs: 20_000,
  detectionDelayMs: 160,
};

export const RECOGNITION_RULES = {
  // Ported from the OpenCV YuNet + SFace Python station profile.
  // For L2-normalized SFace descriptors, cosine 0.36 is roughly distance 1.13.
  matchDistanceThreshold: 1.13,
  matchSimilarityThreshold: 0.36,
  stableFramesRequired: 5,
  faceStateTimeoutMs: 2_000,
  attendanceCooldownMs: 60_000,
  unknownSaveCooldownMs: 60_000,
  maxFacesPerFrame: 12,
  sameFaceCenterRatio: 0.55,
  unknownSuppressionCenterRatio: 0.75,
  unknownSuppressionIoU: 0.10,
  detectionDelayMs: 220,
};
