import { UPLOAD_ACCEPTED_IMAGES, ENROLLMENT_APP_BUILD, QUALITY_RULES } from "./config/appConfig.js";
import { CAPTURE_STEPS } from "./config/captureSteps.js";
import { sleep } from "./utils/async.js";
export class EnrollmentController {
    constructor(view, input, camera, detector, quality, embedder, api, speech) {
        this.view = view;
        this.input = input;
        this.camera = camera;
        this.detector = detector;
        this.quality = quality;
        this.embedder = embedder;
        this.api = api;
        this.speech = speech;
    }
    bind() {
        this.view.startCameraBtn.addEventListener("click", () => void this.startCamera());
        this.view.stopCameraBtn.addEventListener("click", () => this.stopCamera());
        this.view.captureBtn.addEventListener("click", () => void this.captureEnrollment());
    }
    async startCamera() {
        try {
            this.view.setStatus("Loading face detector, landmark model, and SFace model...");
            await Promise.all([this.detector.load(), this.embedder.load()]);
            await this.camera.start();
            this.view.setCameraRunning(true);
            this.view.setStatus(`Camera started. SFace embeddings-only build: ${ENROLLMENT_APP_BUILD}. Put your face inside the green contour.`);
            this.speech.speak("Camera started. Put your face inside the camera frame.", true);
        }
        catch (err) {
            const error = err;
            console.error(error);
            this.view.setStatus(`Camera error: ${error.name}: ${error.message}`);
            this.speech.failure("Camera error. Please check permissions and HTTPS.");
        }
    }
    stopCamera() {
        this.camera.stop();
        this.speech.stop();
        this.view.setCameraRunning(false);
        this.view.setStatus("Camera stopped.");
    }
    async captureEnrollment() {
        this.view.setCaptureEnabled(false);
        this.view.clearPreviews();
        try {
            const identity = this.input.validateIdentity();
            await Promise.all([this.detector.load(), this.embedder.load()]);
            const embeddings = [];
            for (let i = 0; i < CAPTURE_STEPS.length; i++) {
                const embedding = await this.captureStep(identity, CAPTURE_STEPS[i], i);
                embeddings.push(embedding);
                this.speech.success(`${CAPTURE_STEPS[i].label} captured.`);
                await sleep(500);
            }
            this.view.setStep("Uploading SFace embedding file...");
            this.view.setStatus(`Uploading one JSON file with all local SFace embeddings. Build: ${ENROLLMENT_APP_BUILD}`);
            this.speech.speak("Uploading embeddings.", true);
            await this.api.uploadEmbeddingFile(identity, embeddings);
            const summary = await this.api.completeEnrollment(identity);
            this.view.setStep("Enrollment complete.");
            this.view.setStatus(`Done. Images: ${summary.savedImages}, embedding files: ${summary.savedEmbeddingFiles}. Saved to: ${summary.uploadDir}`);
            this.speech.success("Enrollment completed successfully.");
        }
        catch (err) {
            const error = err;
            console.error(error);
            this.view.setStatus(`Error: ${error.message}`);
            this.speech.failure("Enrollment failed. Please try again.");
        }
        finally {
            this.view.setCaptureEnabled(true);
        }
    }
    async captureStep(identity, step, stepIndex) {
        this.view.setStep(`${step.label}: ${step.instruction}`);
        this.speech.speak(step.speech, true);
        const timeoutAt = Date.now() + QUALITY_RULES.captureTimeoutMs;
        let stableGoodFrames = 0;
        let lastResult = null;
        let lastQuality = null;
        while (Date.now() < timeoutAt) {
            const result = await this.detector.detect(this.view.video);
            if (!result) {
                stableGoodFrames = 0;
                this.view.setStatus("No face detected. Put your face inside the contour.");
                this.speech.speak("No face detected. Put your face inside the camera frame.");
                await sleep(150);
                continue;
            }
            const frameCanvas = this.camera.captureFrame(this.view.canvas);
            const quality = this.quality.evaluate(result, frameCanvas);
            const qualityOk = this.quality.isQualityOk(quality);
            const poseOk = step.isPoseOk(quality);
            if (qualityOk && poseOk) {
                stableGoodFrames++;
                this.view.setStatus(`Good. Hold still... ${stableGoodFrames}/${QUALITY_RULES.stableFramesRequired}`);
            }
            else {
                stableGoodFrames = 0;
                const message = !qualityOk ? this.quality.message(quality) : step.instruction;
                this.view.setStatus(message);
                this.speech.speak(message);
            }
            lastResult = result;
            lastQuality = quality;
            if (stableGoodFrames >= QUALITY_RULES.stableFramesRequired)
                break;
            await sleep(QUALITY_RULES.detectionDelayMs);
        }
        if (!lastResult || !lastQuality || stableGoodFrames < QUALITY_RULES.stableFramesRequired) {
            throw new Error(`Could not capture a clear image for: ${step.label}`);
        }
        const frameCanvas = this.camera.captureFrame(this.view.canvas);
        const imageBlob = await this.canvasToBlob(frameCanvas);
        this.view.addPreview(imageBlob, step.label);
        if (UPLOAD_ACCEPTED_IMAGES)
            await this.api.uploadImage(identity, imageBlob, stepIndex + 1, step.pose);
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
    canvasToBlob(sourceCanvas) {
        return new Promise((resolve, reject) => {
            sourceCanvas.toBlob((blob) => {
                if (!blob)
                    return reject(new Error("Could not create image blob."));
                resolve(blob);
            }, "image/jpeg", 0.92);
        });
    }
}
