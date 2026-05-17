import { RECOGNITION_RULES } from "../../shared/appConfig";
import { CameraService } from "../../infrastructure/browser/CameraService";
import { FaceDetectionService } from "../../infrastructure/ml/FaceDetectionService";
import { SFaceEmbeddingService } from "../../infrastructure/ml/SFaceEmbeddingService";
import { RecognitionApi } from "../../infrastructure/api/RecognitionApi";
import { RecognitionView } from "../../presentation/view/RecognitionView";
import { sleep } from "../../shared/async";

export class RecognitionController {
  private running = false;

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
  }

  async start(): Promise<void> {
    try {
      this.running = true;
      this.view.setCameraRunning(true);
      this.view.setStatus("Loading detector, SFace model, and enrolled embeddings...");
      await Promise.all([
        this.detector.load(),
        this.embedder.load(),
        this.refreshKnownFaces(),
      ]);

      await this.camera.start();
      this.view.setStatus("Recognition camera started.");
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
        const result = await this.detector.detect(this.view.video);

        if (!result) {
          this.view.drawOverlay(null, null);
          this.view.setStatus("No face detected.");
          this.view.setResult(null);
          await sleep(RECOGNITION_RULES.detectionDelayMs);
          continue;
        }

        const frameCanvas = this.camera.captureFrame(this.view.canvas);
        const descriptor = await this.embedder.extract(frameCanvas, result);
        const recognition = this.api.recognize(descriptor);

        this.view.drawOverlay(result, recognition);
        this.view.setResult(recognition);
        this.view.setStatus(
          recognition.matched
            ? `Recognized ${recognition.best?.name}.`
            : "Face detected, but no confident match.",
        );
      } catch (err) {
        const error = err as Error;
        console.error(error);
        this.view.setStatus(`Recognition loop error: ${error.message}`);
      }

      await sleep(RECOGNITION_RULES.detectionDelayMs);
    }
  }
}
