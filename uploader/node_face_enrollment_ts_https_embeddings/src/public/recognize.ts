declare const faceapi: any;

const FACE_API_MODEL_URL = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/";
const RECOGNITION_THRESHOLD = 0.50;
const RECOGNITION_INTERVAL_MS = 350;

const video = document.getElementById("video") as HTMLVideoElement;
const startCameraBtn = document.getElementById("startCameraBtn") as HTMLButtonElement;
const stopCameraBtn = document.getElementById("stopCameraBtn") as HTMLButtonElement;
const reloadBtn = document.getElementById("reloadBtn") as HTMLButtonElement;
const statusEl = document.getElementById("status") as HTMLParagraphElement;
const resultBox = document.getElementById("resultBox") as HTMLDivElement;
const resultName = resultBox.querySelector(".result-name") as HTMLDivElement;
const resultDetails = document.getElementById("resultDetails") as HTMLDivElement;
const peopleList = document.getElementById("peopleList") as HTMLDivElement;

let stream: MediaStream | null = null;
let modelsLoaded = false;
let recognitionTimer: number | null = null;

interface StoredCapture {
  pose: string;
  label: string;
  descriptor: number[];
}

interface StoredPerson {
  name: string;
  file: string;
  captures: StoredCapture[];
}

interface EmbeddingApiResponse {
  ok: boolean;
  uploadDir: string;
  people: StoredPerson[];
}

interface FlatDescriptor {
  name: string;
  sourceFile: string;
  pose: string;
  descriptor: Float32Array;
}

let knownDescriptors: FlatDescriptor[] = [];

function setStatus(message: string): void {
  statusEl.textContent = message;
}

function euclideanDistance(a: Float32Array, b: Float32Array): number {
  let sum = 0;

  for (let i = 0; i < a.length; i += 1) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }

  return Math.sqrt(sum);
}

async function loadModels(): Promise<void> {
  if (modelsLoaded) {
    return;
  }

  if (typeof faceapi === "undefined") {
    throw new Error("face-api library was not loaded.");
  }

  setStatus("Loading face models...");

  await Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri(FACE_API_MODEL_URL),
    faceapi.nets.faceLandmark68Net.loadFromUri(FACE_API_MODEL_URL),
    faceapi.nets.faceRecognitionNet.loadFromUri(FACE_API_MODEL_URL),
  ]);

  modelsLoaded = true;
}

async function loadKnownEmbeddings(): Promise<void> {
  setStatus("Loading saved embeddings...");

  const response = await fetch("/api/embeddings");

  if (!response.ok) {
    throw new Error(`Could not load embeddings. Status ${response.status}`);
  }

  const data = (await response.json()) as EmbeddingApiResponse;

  knownDescriptors = [];

  for (const person of data.people) {
    for (const capture of person.captures) {
      if (!Array.isArray(capture.descriptor)) {
        continue;
      }

      knownDescriptors.push({
        name: person.name,
        sourceFile: person.file,
        pose: capture.pose,
        descriptor: new Float32Array(capture.descriptor),
      });
    }
  }

  peopleList.innerHTML = "";

  if (data.people.length === 0) {
    peopleList.innerHTML = "<div>No embedding files found yet.</div>";
  } else {
    for (const person of data.people) {
      const item = document.createElement("div");
      item.textContent = `${person.name} — ${person.captures.length} descriptor(s) — ${person.file}`;
      peopleList.appendChild(item);
    }
  }

  setStatus(`Loaded ${knownDescriptors.length} descriptor(s) from ${data.people.length} person file(s).`);
}

function recognizeDescriptor(descriptor: Float32Array): { name: string; distance: number; pose: string } {
  let bestName = "Unknown";
  let bestPose = "";
  let bestDistance = Number.POSITIVE_INFINITY;

  for (const known of knownDescriptors) {
    if (known.descriptor.length !== descriptor.length) {
      continue;
    }

    const distance = euclideanDistance(descriptor, known.descriptor);

    if (distance < bestDistance) {
      bestDistance = distance;
      bestName = known.name;
      bestPose = known.pose;
    }
  }

  if (bestDistance > RECOGNITION_THRESHOLD) {
    return {
      name: "Unknown",
      distance: bestDistance,
      pose: bestPose,
    };
  }

  return {
    name: bestName,
    distance: bestDistance,
    pose: bestPose,
  };
}

function showRecognition(name: string, distance: number, extra: string): void {
  const isKnown = name !== "Unknown";

  resultBox.classList.toggle("known", isKnown);
  resultBox.classList.toggle("unknown", !isKnown);

  resultName.textContent = name;
  resultDetails.textContent = `Distance: ${Number.isFinite(distance) ? distance.toFixed(3) : "N/A"} ${extra}`;
}

async function recognitionLoop(): Promise<void> {
  if (!stream || !modelsLoaded || knownDescriptors.length === 0) {
    return;
  }

  if (!video.videoWidth || !video.videoHeight) {
    return;
  }

  const result = await faceapi
    .detectSingleFace(
      video,
      new faceapi.TinyFaceDetectorOptions({
        inputSize: 320,
        scoreThreshold: 0.5,
      })
    )
    .withFaceLandmarks()
    .withFaceDescriptor();

  if (!result) {
    showRecognition("Unknown", Number.POSITIVE_INFINITY, "No face detected.");
    return;
  }

  const recognition = recognizeDescriptor(result.descriptor as Float32Array);

  showRecognition(
    recognition.name,
    recognition.distance,
    recognition.name === "Unknown" ? "No match." : `Matched pose: ${recognition.pose}`
  );
}

async function startRecognition(): Promise<void> {
  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("Camera API is not available. Use HTTPS or localhost.");
    }

    await loadModels();
    await loadKnownEmbeddings();

    if (knownDescriptors.length === 0) {
      throw new Error("No local embedding JSON files were found. Enroll a person first.");
    }

    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    });

    video.srcObject = stream;

    startCameraBtn.disabled = true;
    stopCameraBtn.disabled = false;

    setStatus("Recognition started.");

    if (recognitionTimer !== null) {
      window.clearInterval(recognitionTimer);
    }

    recognitionTimer = window.setInterval(() => {
      void recognitionLoop();
    }, RECOGNITION_INTERVAL_MS);
  } catch (err) {
    const error = err as Error;
    console.error(error);
    setStatus(`Error: ${error.message}`);
  }
}

function stopRecognition(): void {
  if (recognitionTimer !== null) {
    window.clearInterval(recognitionTimer);
    recognitionTimer = null;
  }

  if (stream) {
    for (const track of stream.getTracks()) {
      track.stop();
    }
  }

  stream = null;
  video.srcObject = null;

  startCameraBtn.disabled = false;
  stopCameraBtn.disabled = true;

  setStatus("Recognition stopped.");
}

startCameraBtn.addEventListener("click", () => void startRecognition());
stopCameraBtn.addEventListener("click", stopRecognition);
reloadBtn.addEventListener("click", () => void loadKnownEmbeddings());

void loadKnownEmbeddings().catch((err: Error) => {
  console.warn(err);
  setStatus("No embeddings loaded yet. Enroll a person first.");
});
