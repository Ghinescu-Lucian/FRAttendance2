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

const BASE_RECOGNITION_RULES = {
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
  saveUnknownFaces: true,
  captureFrameIntervalMs: 300,
  captureIdleMs: 2500,
  captureMaxQueuedFrames: 90,
  captureBulkConcurrency: 3,
};

const RECOGNITION_PROFILE_OVERRIDES: Record<string, Partial<typeof BASE_RECOGNITION_RULES>> = {
  fast_short: {
    matchDistanceThreshold: 1.13,
    matchSimilarityThreshold: 0.36,
    maxFacesPerFrame: 12,
    detectionDelayMs: 220,
    saveUnknownFaces: true,
  },
  many_faces_unknown: {
    matchDistanceThreshold: 1.13,
    matchSimilarityThreshold: 0.36,
    maxFacesPerFrame: 14,
    detectionDelayMs: 210,
    saveUnknownFaces: true,
  },
  fast_clean: {
    matchDistanceThreshold: 1.13,
    matchSimilarityThreshold: 0.36,
    maxFacesPerFrame: 10,
    detectionDelayMs: 220,
    saveUnknownFaces: false,
  },
  high_recall_many_faces: {
    matchDistanceThreshold: 1.10,
    matchSimilarityThreshold: 0.38,
    maxFacesPerFrame: 20,
    detectionDelayMs: 130,
    unknownSaveCooldownMs: 45_000,
    captureFrameIntervalMs: 250,
    captureIdleMs: 2200,
    captureMaxQueuedFrames: 120,
    captureBulkConcurrency: 3,
    saveUnknownFaces: true,
  },
  multi_attendance_zoom: {
    matchDistanceThreshold: 1.10,
    matchSimilarityThreshold: 0.38,
    maxFacesPerFrame: 12,
    detectionDelayMs: 200,
    saveUnknownFaces: true,
  },
  entrance_mode: {
    matchDistanceThreshold: 1.10,
    matchSimilarityThreshold: 0.38,
    maxFacesPerFrame: 10,
    detectionDelayMs: 120,
    unknownSaveCooldownMs: 45_000,
    captureFrameIntervalMs: 220,
    captureIdleMs: 2000,
    captureMaxQueuedFrames: 120,
    captureBulkConcurrency: 3,
    saveUnknownFaces: true,
  },
};

export const SELECTED_RECOGNITION_PROFILE = moodleContext?.stationProfile || "fast_short";
export const RECOGNITION_RULES = {
  ...BASE_RECOGNITION_RULES,
  ...(RECOGNITION_PROFILE_OVERRIDES[SELECTED_RECOGNITION_PROFILE] || RECOGNITION_PROFILE_OVERRIDES.fast_short),
};

export const FACE_DETECTION_OPTIONS = {
  inputSize: SELECTED_RECOGNITION_PROFILE === "fast_short" ? 416 : 512,
  scoreThreshold: SELECTED_RECOGNITION_PROFILE === "high_recall_many_faces" ? 0.35 : 0.5,
};
