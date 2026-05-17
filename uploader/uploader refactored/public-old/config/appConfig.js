export const FACE_API_MODEL_URL = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/";
export const SFACE_MODEL_URL = "/models/face_recognition_sface_2021dec.onnx";
export const ORT_WASM_PATH = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
export const ENROLLMENT_APP_BUILD = "sface-modular-speech-2026-05-10";
export const UPLOAD_ACCEPTED_IMAGES = false;
export const SFACE_SIZE = 112;
export const SFACE_REFERENCE_POINTS = [
    { x: 38.2946, y: 51.6963 },
    { x: 73.5318, y: 51.5014 },
    { x: 56.0252, y: 71.7366 },
    { x: 41.5493, y: 92.3655 },
    { x: 70.7299, y: 92.2041 },
];
export const QUALITY_RULES = {
    minDetectionScore: 0.5,
    minBlur: 45,
    minBrightness: 45,
    maxBrightness: 225,
    minFaceHeightRatio: 0.18,
    maxFaceHeightRatio: 0.7,
    stableFramesRequired: 3,
    captureTimeoutMs: 20000,
    detectionDelayMs: 160,
};
