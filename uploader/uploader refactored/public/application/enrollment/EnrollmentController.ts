import { UPLOAD_ACCEPTED_IMAGES, ENROLLMENT_APP_BUILD, QUALITY_RULES } from "../../shared/appConfig";
import { CAPTURE_STEPS } from "../../domain/captureSteps";
import { EnrollmentApi } from "../../infrastructure/api/EnrollmentApi";
import { CameraService } from "../../infrastructure/browser/CameraService";
import { FaceDetectionService } from "../../infrastructure/ml/FaceDetectionService";
import { InputService } from "./InputService";
import { QualityService } from "./QualityService";
import { SFaceEmbeddingService } from "../../infrastructure/ml/SFaceEmbeddingService";
import { SpeechService } from "../../infrastructure/browser/SpeechService";
import { CapturedEmbedding, CaptureStep, StudentIdentity } from "../../domain/types";
import { EnrollmentView } from "../../presentation/view/EnrollmentView";
import { sleep } from "../../shared/async";

export class EnrollmentController {
  constructor(
    private readonly view: EnrollmentView,
    private readonly input: InputService,
    private readonly camera: CameraService,
    private readonly detector: FaceDetectionService,
    private readonly quality: QualityService,
    private readonly embedder: SFaceEmbeddingService,
    private readonly api: EnrollmentApi,
    private readonly speech: SpeechService,
  ) {}

  bind(): void {
    this.view.startCameraBtn.addEventListener("click", () => void this.startCamera());
    this.view.stopCameraBtn.addEventListener("click", () => this.stopCamera());
    this.view.captureBtn.addEventListener("click", () => void this.captureEnrollment());
  }

  async startCamera(): Promise<void> {
    try {
      this.view.setStatus("Loading face detector, landmark model, and SFace model...");
      await Promise.all([this.detector.load(), this.embedder.load()]);
      await this.camera.start();
      this.view.setCameraRunning(true);
      this.view.startGuidance(CAPTURE_STEPS[0]);
      this.view.setStatus(`Camera started. SFace embeddings-only build: ${ENROLLMENT_APP_BUILD}. Put your face inside the green contour.`);
    } catch (err) {
      const error = err as Error;
      console.error(error);
      this.view.setStatus(`Camera error: ${error.name}: ${error.message}`);
      this.speech.failure("Camera error. Please check permissions and HTTPS.");
    }
  }

  stopCamera(): void {
    this.camera.stop();
    this.speech.stop();
    this.view.clearOverlay();
    this.view.setCameraRunning(false);
    this.view.setStatus("Camera stopped.");
  }

  async captureEnrollment(): Promise<void> {
    this.view.setCaptureEnabled(false);
    this.view.clearPreviews();

    try {
      const identity = this.input.validateIdentity();
      await Promise.all([this.detector.load(), this.embedder.load()]);

      const embeddings: CapturedEmbedding[] = [];
      for (let i = 0; i < CAPTURE_STEPS.length; i++) {
        const embedding = await this.captureStep(identity, CAPTURE_STEPS[i], i);
        embeddings.push(embedding);
        this.speech.success(`${CAPTURE_STEPS[i].label} captured.`);
        await sleep(500);
      }

      this.view.setStep("Uploading SFace embedding file...");
      this.view.setStatus(`Uploading one JSON file with all local SFace embeddings. Build: ${ENROLLMENT_APP_BUILD}`);
      await this.api.uploadEmbeddingFile(identity, embeddings);

      const summary = await this.api.completeEnrollment(identity);
      this.view.setStep("Enrollment complete.");
      this.view.setStatus(`Done. Images: ${summary.savedImages}, embedding files: ${summary.savedEmbeddingFiles}. Saved to: ${summary.uploadDir}`);
      this.view.clearOverlay();
      this.speech.success("Enrollment completed successfully.");
    } catch (err) {
      const error = err as Error;
      console.error(error);
      this.view.setStatus(`Error: ${error.message}`);
      this.speech.failure("Enrollment failed. Please try again.");
    } finally {
      this.view.setCaptureEnabled(true);
    }
  }

  private async captureStep(identity: StudentIdentity, step: CaptureStep, stepIndex: number): Promise<CapturedEmbedding> {
    this.view.setStep(`${step.label}: ${step.instruction}`);
    this.view.startGuidance(step);
    this.speech.setInstruction(step.instruction, `${step.pose}:pose`);

    const timeoutAt = Date.now() + QUALITY_RULES.captureTimeoutMs;
    let stableGoodFrames = 0;
    let lastResult: any | null = null;
    let lastQuality: CapturedEmbedding["quality"] | null = null;

    while (Date.now() < timeoutAt) {
      const result = await this.detector.detect(this.view.video);
      this.view.updateFaceDetection(result);

      if (!result) {
        stableGoodFrames = 0;
        const message = "No face detected. Put your face inside the camera frame.";
        const key = `${step.pose}:no-face`;
        this.view.setStatus(message);
        this.speech.remindIfIgnored(message, key);
        await sleep(150);
        continue;
      }

      const frameCanvas = this.camera.captureFrame(this.view.canvas);
      const quality = this.quality.evaluate(result, frameCanvas);
      const qualityOk = this.quality.isQualityOk(quality);
      const poseOk = step.isPoseOk(quality);

      if (qualityOk && poseOk) {
        stableGoodFrames++;
        this.speech.setInstruction("Hold still.", `${step.pose}:good`);
        this.view.setStatus(`Good. Hold still... ${stableGoodFrames}/${QUALITY_RULES.stableFramesRequired}`);
      } else {
        stableGoodFrames = 0;
        const message = !qualityOk ? this.quality.message(quality) : step.instruction;
        const key = !qualityOk ? `${step.pose}:quality:${message}` : `${step.pose}:pose`;
        this.view.setStatus(message);
        this.speech.remindIfIgnored(message, key);
      }

      lastResult = result;
      lastQuality = quality;

      if (stableGoodFrames >= QUALITY_RULES.stableFramesRequired) break;
      await sleep(QUALITY_RULES.detectionDelayMs);
    }

    if (!lastResult || !lastQuality || stableGoodFrames < QUALITY_RULES.stableFramesRequired) {
      throw new Error(`Could not capture a clear image for: ${step.label}`);
    }

    const frameCanvas = this.camera.captureFrame(this.view.canvas);
    const imageBlob = await this.canvasToBlob(frameCanvas);
    this.view.addPreview(imageBlob, step.label);

    if (UPLOAD_ACCEPTED_IMAGES) await this.api.uploadImage(identity, imageBlob, stepIndex + 1, step.pose);

    this.view.setStatus(`Generating SFace embedding for ${step.label}...`);
    const descriptor = await this.embedder.extract(frameCanvas, lastResult);

    return {
      studentId: identity.studentId,
      name: identity.personName,
      pose: step.pose,
      label: step.label,
      descriptor,
      quality: lastQuality,
      capturedAt: new Date().toISOString(),
    };
  }

  private canvasToBlob(sourceCanvas: HTMLCanvasElement): Promise<Blob> {
    return new Promise((resolve, reject) => {
      sourceCanvas.toBlob((blob) => {
        if (!blob) return reject(new Error("Could not create image blob."));
        resolve(blob);
      }, "image/jpeg", 0.92);
    });
  }
}
