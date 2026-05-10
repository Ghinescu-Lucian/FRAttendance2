declare const faceapi: any;
declare const ort: any;

const FACE_API_MODEL_URL = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/";

// Put this file in your web app public/models folder so it is served from this URL.
// Use the same model as the Python attendance app:
// models/face_recognition_sface_2021dec.onnx
const SFACE_MODEL_URL = "/models/face_recognition_sface_2021dec.onnx";

const UPLOAD_ACCEPTED_IMAGES = true; // set to false if you want only the single JSON embeddings file

const video = document.getElementById("video") as HTMLVideoElement;
const canvas = document.getElementById("canvas") as HTMLCanvasElement;
const startCameraBtn = document.getElementById("startCameraBtn") as HTMLButtonElement;
const captureBtn = document.getElementById("captureBtn") as HTMLButtonElement;
const stopCameraBtn = document.getElementById("stopCameraBtn") as HTMLButtonElement;
const studentIdInput = document.getElementById("studentId") as HTMLInputElement;
const personNameInput = document.getElementById("personName") as HTMLInputElement;
const statusEl = document.getElementById("status") as HTMLParagraphElement;
const stepEl = document.getElementById("step") as HTMLParagraphElement;
const previewsEl = document.getElementById("previews") as HTMLDivElement;

let stream: MediaStream | null = null;
let faceApiModelsLoaded = false;
let sfaceSession: any | null = null;

interface QualityInfo {
  score: number;
  blur: number;
  brightness: number;
  faceHeightRatio: number;
  centered: boolean;
  yaw: number;
  pitch: number;
}

interface CapturedEmbedding {
  studentId: string;
  name: string;
  pose: string;
  label: string;
  descriptor: number[];
  quality: QualityInfo;
  capturedAt: string;
}

interface CaptureStep {
  pose: string;
  label: string;
  instruction: string;
  isPoseOk: (quality: QualityInfo) => boolean;
}

interface Point2D {
  x: number;
  y: number;
}

const CAPTURE_STEPS: CaptureStep[] = [
  { pose: "front", label: "Front", instruction: "Look straight at the camera.", isPoseOk: (q) => Math.abs(q.yaw) < 0.12 && Math.abs(q.pitch) < 0.12 },
  { pose: "right", label: "Head right", instruction: "Turn your head slightly to YOUR right.", isPoseOk: (q) => q.yaw > 0.10 },
  { pose: "left", label: "Head left", instruction: "Turn your head slightly to YOUR left.", isPoseOk: (q) => q.yaw < -0.10 },
  { pose: "down", label: "Head down", instruction: "Tilt your head slightly down.", isPoseOk: (q) => q.pitch > 0.06 },
  { pose: "front_2", label: "Front again", instruction: "Look straight again.", isPoseOk: (q) => Math.abs(q.yaw) < 0.12 && Math.abs(q.pitch) < 0.12 },
];

// OpenCV FaceRecognizerSF alignCrop uses these 5 target landmark positions and a 112x112 aligned face.
const SFACE_SIZE = 112;
const SFACE_REFERENCE_POINTS: Point2D[] = [
  { x: 38.2946, y: 51.6963 },
  { x: 73.5318, y: 51.5014 },
  { x: 56.0252, y: 71.7366 },
  { x: 41.5493, y: 92.3655 },
  { x: 70.7299, y: 92.2041 },
];

function setStatus(message: string): void { statusEl.textContent = message; }
function setStep(message: string): void { stepEl.textContent = message; }
function sleep(ms: number): Promise<void> { return new Promise((resolve) => window.setTimeout(resolve, ms)); }

function validateInputs(): { studentId: string; personName: string } {
  const studentId = studentIdInput.value.trim();
  const personName = personNameInput.value.trim();
  if (!studentId) throw new Error("Student ID is required.");
  if (!personName) throw new Error("Person name is required.");
  return { studentId, personName };
}

async function loadFaceApiModels(): Promise<void> {
  if (faceApiModelsLoaded) return;
  if (typeof faceapi === "undefined") throw new Error("face-api library was not loaded.");

  setStatus("Loading face detector and landmark models...");
  await Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri(FACE_API_MODEL_URL),
    faceapi.nets.faceLandmark68Net.loadFromUri(FACE_API_MODEL_URL),
  ]);

  faceApiModelsLoaded = true;
}

async function loadSFaceModel(): Promise<void> {
  if (sfaceSession) return;
  if (typeof ort === "undefined") throw new Error("onnxruntime-web was not loaded. Add ort.min.js in index.html.");

  // Needed when using the CDN script for onnxruntime-web.
  // Remove or change this if you bundle onnxruntime-web locally.
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

  setStatus("Loading OpenCV SFace ONNX model...");
  sfaceSession = await ort.InferenceSession.create(SFACE_MODEL_URL, {
    executionProviders: ["wasm"],
  });
}

async function loadModels(): Promise<void> {
  await loadFaceApiModels();
  await loadSFaceModel();
}

async function startCamera(): Promise<void> {
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("Camera API is not available. Use HTTPS or localhost.");
    await loadModels();
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
    video.srcObject = stream;
    startCameraBtn.disabled = true;
    captureBtn.disabled = false;
    stopCameraBtn.disabled = false;
    setStatus("Camera started. Put your face inside the green contour.");
  } catch (err) {
    const error = err as Error;
    console.error(error);
    setStatus(`Camera error: ${error.name}: ${error.message}`);
  }
}

function stopCamera(): void {
  if (stream) for (const track of stream.getTracks()) track.stop();
  stream = null;
  video.srcObject = null;
  startCameraBtn.disabled = false;
  captureBtn.disabled = true;
  stopCameraBtn.disabled = true;
  setStatus("Camera stopped.");
}

function getDetectionOptions(): any {
  return new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.5 });
}

async function detectFace(): Promise<any | null> {
  if (!video.videoWidth || !video.videoHeight || !faceApiModelsLoaded) return null;
  // Keep face-api only for detection + 68 landmarks + quality/pose checks.
  // The actual recognition descriptor below is OpenCV SFace, not face-api.
  return await faceapi.detectSingleFace(video, getDetectionOptions()).withFaceLandmarks();
}

function averagePoint(points: Point2D[]): Point2D {
  const sum = points.reduce((acc, p) => ({ x: acc.x + p.x, y: acc.y + p.y }), { x: 0, y: 0 });
  return { x: sum.x / points.length, y: sum.y / points.length };
}

function getSFaceSourcePoints(result: any): Point2D[] {
  const landmarks = result.landmarks;
  const leftEye = averagePoint(landmarks.getLeftEye());
  const rightEye = averagePoint(landmarks.getRightEye());
  const nose = landmarks.getNose()[3] || landmarks.getNose()[0];
  const mouth = landmarks.getMouth();
  const leftMouth = mouth[0];
  const rightMouth = mouth[6];
  return [leftEye, rightEye, nose, leftMouth, rightMouth];
}

function estimatePose(result: any): { yaw: number; pitch: number } {
  const landmarks = result.landmarks;
  const leftEye = averagePoint(landmarks.getLeftEye());
  const rightEye = averagePoint(landmarks.getRightEye());
  const nose = landmarks.getNose()[3] || landmarks.getNose()[0];
  const mouth = averagePoint(landmarks.getMouth());
  const eyeMid = { x: (leftEye.x + rightEye.x) / 2, y: (leftEye.y + rightEye.y) / 2 };
  const eyeDistance = Math.max(1, Math.hypot(rightEye.x - leftEye.x, rightEye.y - leftEye.y));
  const vertical = Math.max(1, mouth.y - eyeMid.y);
  const yaw = (nose.x - eyeMid.x) / eyeDistance;
  const pitch = (nose.y - eyeMid.y) / vertical - 0.48;
  return { yaw, pitch };
}

function captureCurrentFrame(): HTMLCanvasElement {
  if (!video.videoWidth || !video.videoHeight) throw new Error("Video is not ready yet.");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Could not create canvas context.");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas;
}

function getImageDataForBox(sourceCanvas: HTMLCanvasElement, box: any): ImageData {
  const ctx = sourceCanvas.getContext("2d");
  if (!ctx) throw new Error("Could not create canvas context.");
  const x = Math.max(0, Math.floor(box.x));
  const y = Math.max(0, Math.floor(box.y));
  const width = Math.max(1, Math.min(sourceCanvas.width - x, Math.floor(box.width)));
  const height = Math.max(1, Math.min(sourceCanvas.height - y, Math.floor(box.height)));
  return ctx.getImageData(x, y, width, height);
}

function brightnessScore(imageData: ImageData): number {
  const data = imageData.data;
  let total = 0;
  const pixels = data.length / 4;
  for (let i = 0; i < data.length; i += 4) total += 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  return total / pixels;
}

function blurScore(imageData: ImageData): number {
  const { width, height, data } = imageData;
  if (width < 3 || height < 3) return 0;
  const gray = new Float32Array(width * height);
  for (let i = 0, j = 0; i < data.length; i += 4, j++) gray[j] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  let sum = 0, sumSq = 0, count = 0;
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const idx = y * width + x;
      const lap = 4 * gray[idx] - gray[idx - 1] - gray[idx + 1] - gray[idx - width] - gray[idx + width];
      sum += lap; sumSq += lap * lap; count++;
    }
  }
  const mean = sum / count;
  return sumSq / count - mean * mean;
}

function isFaceCentered(box: any, frameWidth: number, frameHeight: number): boolean {
  const centerX = box.x + box.width / 2;
  const centerY = box.y + box.height / 2;
  return centerX >= frameWidth * 0.24 && centerX <= frameWidth * 0.76 && centerY >= frameHeight * 0.15 && centerY <= frameHeight * 0.85;
}

function evaluateQuality(result: any): QualityInfo {
  const frameCanvas = captureCurrentFrame();
  const box = result.detection.box;
  const faceImage = getImageDataForBox(frameCanvas, box);
  const pose = estimatePose(result);
  return {
    score: result.detection.score,
    blur: blurScore(faceImage),
    brightness: brightnessScore(faceImage),
    faceHeightRatio: box.height / frameCanvas.height,
    centered: isFaceCentered(box, frameCanvas.width, frameCanvas.height),
    yaw: pose.yaw,
    pitch: pose.pitch,
  };
}

function isQualityOk(q: QualityInfo): boolean {
  return q.score >= 0.5 && q.blur >= 45 && q.brightness >= 45 && q.brightness <= 225 && q.faceHeightRatio >= 0.18 && q.faceHeightRatio <= 0.70 && q.centered;
}

function qualityMessage(q: QualityInfo): string {
  if (q.blur < 45) return "Image is blurry. Hold the phone steady.";
  if (q.brightness < 45) return "Image is too dark. Add light.";
  if (q.brightness > 225) return "Image is too bright.";
  if (q.faceHeightRatio < 0.18) return "Move closer.";
  if (q.faceHeightRatio > 0.70) return "Move slightly farther.";
  if (!q.centered) return "Center your face inside the green contour.";
  return "Good image.";
}

function getSimilarityTransform(src: Point2D[], dst: Point2D[]): [number, number, number, number, number, number] {
  if (src.length !== 5 || dst.length !== 5) throw new Error("SFace alignment needs exactly 5 landmarks.");

  const srcMean = averagePoint(src);
  const dstMean = averagePoint(dst);

  let den = 0;
  let aNum = 0;
  let bNum = 0;

  for (let i = 0; i < src.length; i++) {
    const sx = src[i].x - srcMean.x;
    const sy = src[i].y - srcMean.y;
    const dx = dst[i].x - dstMean.x;
    const dy = dst[i].y - dstMean.y;

    den += sx * sx + sy * sy;
    aNum += sx * dx + sy * dy;
    bNum += sx * dy - sy * dx;
  }

  if (den <= 1e-6) throw new Error("Invalid face landmarks for SFace alignment.");

  const a = aNum / den;
  const b = bNum / den;
  const c = -b;
  const d = a;
  const e = dstMean.x - a * srcMean.x - c * srcMean.y;
  const f = dstMean.y - b * srcMean.x - d * srcMean.y;

  // Canvas setTransform(a, b, c, d, e, f):
  // x' = a*x + c*y + e
  // y' = b*x + d*y + f
  return [a, b, c, d, e, f];
}

function alignFaceForSFace(sourceCanvas: HTMLCanvasElement, result: any): HTMLCanvasElement {
  const srcPoints = getSFaceSourcePoints(result);
  const [a, b, c, d, e, f] = getSimilarityTransform(srcPoints, SFACE_REFERENCE_POINTS);

  const alignedCanvas = document.createElement("canvas");
  alignedCanvas.width = SFACE_SIZE;
  alignedCanvas.height = SFACE_SIZE;

  const ctx = alignedCanvas.getContext("2d");
  if (!ctx) throw new Error("Could not create aligned face canvas context.");

  ctx.setTransform(a, b, c, d, e, f);
  ctx.drawImage(sourceCanvas, 0, 0);
  ctx.setTransform(1, 0, 0, 1, 0, 0);

  return alignedCanvas;
}

function canvasToSFaceInputTensor(alignedCanvas: HTMLCanvasElement): any {
  const ctx = alignedCanvas.getContext("2d");
  if (!ctx) throw new Error("Could not create aligned face canvas context.");

  const imageData = ctx.getImageData(0, 0, SFACE_SIZE, SFACE_SIZE).data;
  const chw = new Float32Array(1 * 3 * SFACE_SIZE * SFACE_SIZE);
  const planeSize = SFACE_SIZE * SFACE_SIZE;

  for (let y = 0; y < SFACE_SIZE; y++) {
    for (let x = 0; x < SFACE_SIZE; x++) {
      const srcIdx = (y * SFACE_SIZE + x) * 4;
      const dstIdx = y * SFACE_SIZE + x;

      // OpenCV FaceRecognizerSF uses blobFromImage(..., scalefactor=1, mean=0, swapRB=true).
      // Since canvas is already RGB, feed RGB channels, values 0..255, CHW layout.
      chw[dstIdx] = imageData[srcIdx];
      chw[planeSize + dstIdx] = imageData[srcIdx + 1];
      chw[2 * planeSize + dstIdx] = imageData[srcIdx + 2];
    }
  }

  return new ort.Tensor("float32", chw, [1, 3, SFACE_SIZE, SFACE_SIZE]);
}

function l2Normalize(values: number[]): number[] {
  let sumSq = 0;
  for (const v of values) sumSq += v * v;
  const norm = Math.sqrt(sumSq);
  if (!Number.isFinite(norm) || norm <= 1e-12) return values;
  return values.map((v) => v / norm);
}

async function extractSFaceDescriptor(frameCanvas: HTMLCanvasElement, result: any): Promise<number[]> {
  if (!sfaceSession) throw new Error("SFace ONNX session is not loaded.");

  const alignedCanvas = alignFaceForSFace(frameCanvas, result);
  const inputTensor = canvasToSFaceInputTensor(alignedCanvas);
  const inputName = sfaceSession.inputNames[0];
  const outputName = sfaceSession.outputNames[0];
  const outputs = await sfaceSession.run({ [inputName]: inputTensor });
  const output = outputs[outputName];
  const raw = Array.from(output.data as Float32Array);

  if (raw.length !== 128) throw new Error(`SFace descriptor length is ${raw.length}, expected 128.`);
  return l2Normalize(raw);
}

function canvasToBlob(sourceCanvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    sourceCanvas.toBlob((blob) => {
      if (!blob) return reject(new Error("Could not create image blob."));
      resolve(blob);
    }, "image/jpeg", 0.92);
  });
}

async function uploadImage(blob: Blob, frameIndex: number, pose: string): Promise<void> {
  const { studentId, personName } = validateInputs();
  const formData = new FormData();
  formData.append("studentId", studentId);
  formData.append("name", personName);
  formData.append("frameIndex", String(frameIndex));
  formData.append("pose", pose);
  formData.append("image", blob, `${pose}_${frameIndex}.jpg`);
  const response = await fetch("/api/upload-face", { method: "POST", body: formData });
  const data = await response.json().catch(() => null);
  if (!response.ok || !data?.ok) throw new Error(data?.error || `Image upload failed with status ${response.status}`);
}

async function uploadEmbeddingFile(embeddings: CapturedEmbedding[]): Promise<void> {
  const { studentId, personName } = validateInputs();
  const payload = {
    version: 2,
    studentId,
    name: personName,
    createdAt: new Date().toISOString(),
    model: {
      family: "opencv",
      detectorForEnrollment: "@vladmandic/face-api tinyFaceDetector + faceLandmark68Net",
      recognizer: "sface",
      recognizerModel: "face_recognition_sface_2021dec.onnx",
      descriptorLength: embeddings[0]?.descriptor.length ?? 0,
      descriptorNormalized: true,
      alignedSize: [SFACE_SIZE, SFACE_SIZE],
      note: "Descriptors are OpenCV SFace-style embeddings. Recognition side must also use OpenCV SFace or a byte-equivalent SFace pipeline.",
    },
    captures: embeddings,
  };

  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const formData = new FormData();
  formData.append("studentId", studentId);
  formData.append("name", personName);
  formData.append("model", "opencv-sface-2021dec");
  formData.append("embeddingFile", blob, `${personName}_${studentId}_sface_embeddings.json`);

  const response = await fetch("/api/upload-embedding-file", { method: "POST", body: formData });
  const data = await response.json().catch(() => null);
  if (!response.ok || !data?.ok) throw new Error(data?.error || `Embedding upload failed with status ${response.status}`);
}

async function completeEnrollment(): Promise<any> {
  const { studentId, personName } = validateInputs();
  const response = await fetch("/api/complete-enrollment", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ studentId, name: personName }),
  });
  if (!response.ok) throw new Error(`Complete enrollment failed with status ${response.status}`);
  return await response.json();
}

function addPreview(blob: Blob, label: string): void {
  const wrapper = document.createElement("div");
  wrapper.className = "preview-item";
  const img = document.createElement("img");
  img.src = URL.createObjectURL(blob);
  const caption = document.createElement("span");
  caption.textContent = label;
  wrapper.appendChild(img);
  wrapper.appendChild(caption);
  previewsEl.prepend(wrapper);
}

async function captureStep(step: CaptureStep, stepIndex: number): Promise<CapturedEmbedding> {
  const { studentId, personName } = validateInputs();

  setStep(`${step.label}: ${step.instruction}`);
  const timeoutAt = Date.now() + 20000;
  let stableGoodFrames = 0;
  let lastResult: any | null = null;
  let lastQuality: QualityInfo | null = null;

  while (Date.now() < timeoutAt) {
    const result = await detectFace();
    if (!result) {
      stableGoodFrames = 0;
      setStatus("No face detected. Put your face inside the contour.");
      await sleep(150);
      continue;
    }

    const quality = evaluateQuality(result);
    const qualityOk = isQualityOk(quality);
    const poseOk = step.isPoseOk(quality);

    if (qualityOk && poseOk) {
      stableGoodFrames++;
      setStatus(`Good. Hold still... ${stableGoodFrames}/3`);
    } else {
      stableGoodFrames = 0;
      setStatus(!qualityOk ? qualityMessage(quality) : step.instruction);
    }

    lastResult = result;
    lastQuality = quality;
    if (stableGoodFrames >= 3) break;
    await sleep(160);
  }

  if (!lastResult || !lastQuality || stableGoodFrames < 3) throw new Error(`Could not capture a clear image for: ${step.label}`);

  const frameCanvas = captureCurrentFrame();
  const imageBlob = await canvasToBlob(frameCanvas);
  addPreview(imageBlob, step.label);

  if (UPLOAD_ACCEPTED_IMAGES) await uploadImage(imageBlob, stepIndex + 1, step.pose);

  setStatus(`Generating SFace embedding for ${step.label}...`);
  const descriptor = await extractSFaceDescriptor(frameCanvas, lastResult);

  return {
    studentId,
    name: personName,
    pose: step.pose,
    label: step.label,
    descriptor,
    quality: lastQuality,
    capturedAt: new Date().toISOString(),
  };
}

async function captureEnrollment(): Promise<void> {
  captureBtn.disabled = true;
  previewsEl.innerHTML = "";

  try {
    validateInputs();
    await loadModels();

    const embeddings: CapturedEmbedding[] = [];
    for (let i = 0; i < CAPTURE_STEPS.length; i++) {
      const embedding = await captureStep(CAPTURE_STEPS[i], i);
      embeddings.push(embedding);
      await sleep(500);
    }

    setStep("Uploading SFace embedding file...");
    setStatus("Uploading one JSON file with all local SFace embeddings...");
    await uploadEmbeddingFile(embeddings);

    const summary = await completeEnrollment();
    setStep("Enrollment complete.");
    setStatus(`Done. Images: ${summary.savedImages}, embedding files: ${summary.savedEmbeddingFiles}. Saved to: ${summary.uploadDir}`);
  } catch (err) {
    const error = err as Error;
    console.error(error);
    setStatus(`Error: ${error.message}`);
  } finally {
    captureBtn.disabled = false;
  }
}

startCameraBtn.addEventListener("click", () => void startCamera());
stopCameraBtn.addEventListener("click", stopCamera);
captureBtn.addEventListener("click", () => void captureEnrollment());
