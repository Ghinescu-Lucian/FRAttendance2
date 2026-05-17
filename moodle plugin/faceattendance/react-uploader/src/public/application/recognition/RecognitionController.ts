import { RECOGNITION_RULES } from "../../shared/appConfig";
import { CameraService } from "../../infrastructure/browser/CameraService";
import { FaceDetectionService } from "../../infrastructure/ml/FaceDetectionService";
import { SFaceEmbeddingService } from "../../infrastructure/ml/SFaceEmbeddingService";
import { RecognitionApi } from "../../infrastructure/api/RecognitionApi";
import { RecognitionView } from "../../presentation/view/RecognitionView";
import { sleep } from "../../shared/async";
import { RecognitionResult } from "../../domain/types";

type FaceBox = {
  x: number;
  y: number;
  width: number;
  height: number;
};

type KnownState = {
  stableCount: number;
  lastSeen: number;
  lastBox: FaceBox | null;
  lastMarkedAt: number;
};

type UnknownTrack = {
  id: number;
  stableCount: number;
  lastSeen: number;
  lastSavedAt: number;
  box: FaceBox;
};

export class RecognitionController {
  private running = false;
  private knownStates = new Map<string, KnownState>();
  private unknownTracks: UnknownTrack[] = [];
  private nextUnknownTrackId = 1;
  private lastSessionRefreshAt = 0;

  constructor(
    private readonly view: RecognitionView,
    private readonly camera: CameraService,
    private readonly detector: FaceDetectionService,
    private readonly embedder: SFaceEmbeddingService,
    private readonly api: RecognitionApi,
  ) {}

  bind(): void {
    this.view.startBtn.addEventListener("click", () => void this.start());
    this.view.stopBtn.addEventListener("click", () => this.stop());
    this.view.refreshBtn.addEventListener("click", () => void this.refreshKnownFaces());
    this.view.cameraSelect.addEventListener("change", () => void this.switchCameraIfRunning());
    void this.refreshCameras();
  }

  private async refreshCameras(activeDeviceId?: string | null): Promise<void> {
    try {
      const devices = await this.camera.listCameras();
      this.view.setCameraOptions(devices, activeDeviceId);
    } catch (err) {
      console.warn("Could not enumerate cameras", err);
    }
  }

  private async switchCameraIfRunning(): Promise<void> {
    if (!this.running) return;

    try {
      this.view.setStatus("Switching camera...");
      this.camera.stop();
      this.knownStates.clear();
      this.unknownTracks = [];
      await this.camera.start(this.view.getSelectedCameraId());
      await this.refreshCameras(this.camera.getActiveDeviceId());
      this.view.setStatus("Camera switched. Recognition continues.");
    } catch (err) {
      const error = err as Error;
      console.error(error);
      this.running = false;
      this.view.setCameraRunning(false);
      this.view.setStatus(`Camera switch error: ${error.name}: ${error.message}`);
    }
  }

  async start(): Promise<void> {
    try {
      this.running = true;
      this.knownStates.clear();
      this.unknownTracks = [];
      this.view.setCameraRunning(true);
      this.view.setStatus("Loading active session, detector, SFace model, and enrolled embeddings...");

      const [session] = await Promise.all([
        this.api.refreshActiveSession(),
        this.detector.load(),
        this.embedder.load(),
        this.refreshKnownFaces(),
      ]);

      if (!session && window.FACEATTENDANCE_CONTEXT?.mode === "station") {
        this.view.setStatus("No active attendance session right now. Create a session whose current time is inside the start/end interval.");
        this.running = false;
        this.view.setCameraRunning(false);
        return;
      }

      await this.camera.start(this.view.getSelectedCameraId());
      await this.refreshCameras(this.camera.getActiveDeviceId());
      this.view.setStatus(session ? `Camera started. Active session: ${session.name}.` : "Recognition camera started.");
      void this.recognitionLoop();
    } catch (err) {
      const error = err as Error;
      console.error(error);
      this.running = false;
      this.view.setCameraRunning(false);
      this.view.setStatus(`Recognition error: ${error.message}`);
    }
  }

  stop(): void {
    this.running = false;
    this.camera.stop();
    this.knownStates.clear();
    this.unknownTracks = [];
    this.view.clearOverlay();
    this.view.setCameraRunning(false);
    this.view.setStatus("Recognition camera stopped.");
    this.view.setResult(null);
  }

  async refreshKnownFaces(): Promise<void> {
    const faces = await this.api.refreshKnownFaces();
    this.view.setKnownFacesCount(faces.length);
  }

  private async recognitionLoop(): Promise<void> {
    while (this.running) {
      try {
        if (window.FACEATTENDANCE_CONTEXT?.mode === "station") {
          const now = Date.now();
          if (now - this.lastSessionRefreshAt > 5000) {
            await this.api.refreshActiveSession();
            this.lastSessionRefreshAt = now;
          }
          if (!this.api.getActiveSession()) {
            this.view.drawOverlays([]);
            this.view.setStatus("No active attendance session. Waiting...");
            await sleep(2000);
            continue;
          }
        }

        const rawResults = await this.detector.detectAll(this.view.video);
        const results = rawResults.slice(0, RECOGNITION_RULES.maxFacesPerFrame);

        if (!results.length) {
          this.cleanupExpiredStates(Date.now());
          this.view.drawOverlays([]);
          this.view.setStatus("No face detected.");
          this.view.setResult(null);
          await sleep(RECOGNITION_RULES.detectionDelayMs);
          continue;
        }

        const frameCanvas = this.camera.captureFrame(this.view.canvas);
        const decisions: Array<{ detection: any; recognition: RecognitionResult; stableCount?: number }> = [];

        for (const detection of results) {
          const descriptor = await this.embedder.extract(frameCanvas, detection);
          const recognition = this.api.recognize(descriptor);
          const box = this.getDetectionBox(detection);

          if (recognition.matched && recognition.best) {
            const stableCount = this.updateKnownState(recognition, box);
            recognition.stableCount = stableCount;
            decisions.push({ detection, recognition, stableCount });

            if (stableCount >= RECOGNITION_RULES.stableFramesRequired) {
              await this.maybeMarkKnown(recognition);
            }
          } else {
            const nearKnown = this.isUnknownNearKnown(box);
            if (!nearKnown) {
              const stableCount = this.updateUnknownTrack(box);
              recognition.stableCount = stableCount;
              decisions.push({ detection, recognition, stableCount });

              if (stableCount >= RECOGNITION_RULES.stableFramesRequired) {
                const thumbnail = this.makeFaceThumbnail(frameCanvas, box);
                await this.maybeSaveUnknown(descriptor, recognition, box, thumbnail);
              }
            }
          }
        }

        this.cleanupExpiredStates(Date.now());
        this.view.drawOverlays(decisions);
        const matchedCount = decisions.filter((item) => item.recognition.matched).length;
        const unknownCount = decisions.length - matchedCount;
        this.view.setResult(decisions[0]?.recognition || null);
        this.view.setStatus(`Detected ${decisions.length} face(s): ${matchedCount} known, ${unknownCount} unknown. Stable frames required: ${RECOGNITION_RULES.stableFramesRequired}.`);
      } catch (err) {
        const error = err as Error;
        console.error(error);
        this.view.setStatus(`Recognition loop error: ${error.message}`);
      }

      await sleep(RECOGNITION_RULES.detectionDelayMs);
    }
  }

  private updateKnownState(recognition: RecognitionResult, box: FaceBox | null): number {
    const key = recognition.best?.studentId;
    if (!key) return 0;

    const now = Date.now();
    const previous = this.knownStates.get(key);
    const shouldContinue = Boolean(previous && now - previous.lastSeen <= RECOGNITION_RULES.faceStateTimeoutMs);
    const stableCount = shouldContinue ? previous!.stableCount + 1 : 1;

    this.knownStates.set(key, {
      stableCount,
      lastSeen: now,
      lastBox: box,
      lastMarkedAt: previous?.lastMarkedAt || 0,
    });

    return stableCount;
  }

  private async maybeMarkKnown(recognition: RecognitionResult): Promise<void> {
    const key = recognition.best?.studentId;
    if (!key) return;

    const state = this.knownStates.get(key);
    if (!state) return;

    const now = Date.now();
    if (now - state.lastMarkedAt < RECOGNITION_RULES.attendanceCooldownMs) return;

    await this.api.markKnownFace(recognition, "browser-camera-station-sface-script-profile");
    state.lastMarkedAt = now;
    this.knownStates.set(key, state);
  }

  private updateUnknownTrack(box: FaceBox | null): number {
    const now = Date.now();
    if (!box) return 1;

    let track = this.unknownTracks.find((candidate) => this.isSamePhysicalFace(candidate.box, box));
    if (!track) {
      track = {
        id: this.nextUnknownTrackId++,
        stableCount: 0,
        lastSeen: 0,
        lastSavedAt: 0,
        box,
      };
      this.unknownTracks.push(track);
    }

    track.stableCount += 1;
    track.lastSeen = now;
    track.box = box;
    return track.stableCount;
  }

  private async maybeSaveUnknown(descriptor: number[], recognition: RecognitionResult, box: FaceBox | null, thumbnail?: string | null): Promise<void> {
    if (!box) return;

    const track = this.unknownTracks.find((candidate) => this.isSamePhysicalFace(candidate.box, box));
    if (!track) return;

    const now = Date.now();
    if (now - track.lastSavedAt < RECOGNITION_RULES.unknownSaveCooldownMs) return;

    await this.api.saveUnknownFace(descriptor, recognition, "browser-camera-station-sface-script-profile", thumbnail);
    track.lastSavedAt = now;
  }

  private makeFaceThumbnail(frameCanvas: HTMLCanvasElement, box: FaceBox | null): string | null {
    if (!box || !frameCanvas.width || !frameCanvas.height) return null;

    const paddingRatio = 0.35;
    const padX = box.width * paddingRatio;
    const padY = box.height * paddingRatio;

    const sourceX = Math.max(0, Math.floor(box.x - padX));
    const sourceY = Math.max(0, Math.floor(box.y - padY));
    const sourceRight = Math.min(frameCanvas.width, Math.ceil(box.x + box.width + padX));
    const sourceBottom = Math.min(frameCanvas.height, Math.ceil(box.y + box.height + padY));
    const sourceWidth = Math.max(1, sourceRight - sourceX);
    const sourceHeight = Math.max(1, sourceBottom - sourceY);

    const maxSide = 280;
    const scale = Math.min(1, maxSide / Math.max(sourceWidth, sourceHeight));
    const targetWidth = Math.max(1, Math.round(sourceWidth * scale));
    const targetHeight = Math.max(1, Math.round(sourceHeight * scale));

    const thumbnailCanvas = document.createElement("canvas");
    thumbnailCanvas.width = targetWidth;
    thumbnailCanvas.height = targetHeight;

    const context = thumbnailCanvas.getContext("2d");
    if (!context) return null;

    context.drawImage(
      frameCanvas,
      sourceX,
      sourceY,
      sourceWidth,
      sourceHeight,
      0,
      0,
      targetWidth,
      targetHeight,
    );

    return thumbnailCanvas.toDataURL("image/jpeg", 0.78);
  }

  private isUnknownNearKnown(box: FaceBox | null): boolean {
    if (!box) return false;

    for (const state of this.knownStates.values()) {
      if (!state.lastBox) continue;
      if (this.intersectionOverUnion(box, state.lastBox) >= RECOGNITION_RULES.unknownSuppressionIoU) return true;
      if (this.centerDistanceRatio(box, state.lastBox) <= RECOGNITION_RULES.unknownSuppressionCenterRatio) return true;
    }

    return false;
  }

  private cleanupExpiredStates(now: number): void {
    for (const [key, state] of Array.from(this.knownStates.entries())) {
      if (now - state.lastSeen > RECOGNITION_RULES.faceStateTimeoutMs) {
        this.knownStates.delete(key);
      }
    }

    this.unknownTracks = this.unknownTracks.filter((track) => now - track.lastSeen <= RECOGNITION_RULES.faceStateTimeoutMs);
  }

  private getDetectionBox(detection: any): FaceBox | null {
    const box = detection?.detection?.box;
    if (!box) return null;
    return {
      x: Number(box.x) || 0,
      y: Number(box.y) || 0,
      width: Number(box.width) || 0,
      height: Number(box.height) || 0,
    };
  }

  private isSamePhysicalFace(a: FaceBox, b: FaceBox): boolean {
    return this.intersectionOverUnion(a, b) >= 0.3 || this.centerDistanceRatio(a, b) <= RECOGNITION_RULES.sameFaceCenterRatio;
  }

  private intersectionOverUnion(a: FaceBox, b: FaceBox): number {
    const ax2 = a.x + a.width;
    const ay2 = a.y + a.height;
    const bx2 = b.x + b.width;
    const by2 = b.y + b.height;

    const ix1 = Math.max(a.x, b.x);
    const iy1 = Math.max(a.y, b.y);
    const ix2 = Math.min(ax2, bx2);
    const iy2 = Math.min(ay2, by2);

    const interWidth = Math.max(0, ix2 - ix1);
    const interHeight = Math.max(0, iy2 - iy1);
    const interArea = interWidth * interHeight;
    if (interArea <= 0) return 0;

    const aArea = Math.max(0, a.width) * Math.max(0, a.height);
    const bArea = Math.max(0, b.width) * Math.max(0, b.height);
    const unionArea = aArea + bArea - interArea;
    return unionArea > 0 ? interArea / unionArea : 0;
  }

  private centerDistanceRatio(a: FaceBox, b: FaceBox): number {
    const ax = a.x + a.width / 2;
    const ay = a.y + a.height / 2;
    const bx = b.x + b.width / 2;
    const by = b.y + b.height / 2;
    const distance = Math.hypot(ax - bx, ay - by);
    const reference = Math.max(a.width, a.height, b.width, b.height, 1);
    return distance / reference;
  }
}
