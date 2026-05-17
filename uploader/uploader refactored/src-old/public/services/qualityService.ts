import { QUALITY_RULES } from "../config/appConfig.js";
import { Point2D, QualityInfo } from "../types.js";
import { averagePoint } from "../utils/math.js";

export class QualityService {
  evaluate(result: any, frameCanvas: HTMLCanvasElement): QualityInfo {
    const box = result.detection.box;
    const faceImage = this.getImageDataForBox(frameCanvas, box);
    const pose = this.estimatePose(result);

    return {
      score: result.detection.score,
      blur: this.blurScore(faceImage),
      brightness: this.brightnessScore(faceImage),
      faceHeightRatio: box.height / frameCanvas.height,
      centered: this.isFaceCentered(box, frameCanvas.width, frameCanvas.height),
      yaw: pose.yaw,
      pitch: pose.pitch,
    };
  }

  isQualityOk(q: QualityInfo): boolean {
    return q.score >= QUALITY_RULES.minDetectionScore &&
      q.blur >= QUALITY_RULES.minBlur &&
      q.brightness >= QUALITY_RULES.minBrightness &&
      q.brightness <= QUALITY_RULES.maxBrightness &&
      q.faceHeightRatio >= QUALITY_RULES.minFaceHeightRatio &&
      q.faceHeightRatio <= QUALITY_RULES.maxFaceHeightRatio &&
      q.centered;
  }

  message(q: QualityInfo): string {
    if (q.blur < QUALITY_RULES.minBlur) return "Image is blurry. Hold the phone steady.";
    if (q.brightness < QUALITY_RULES.minBrightness) return "Image is too dark. Add light.";
    if (q.brightness > QUALITY_RULES.maxBrightness) return "Image is too bright.";
    if (q.faceHeightRatio < QUALITY_RULES.minFaceHeightRatio) return "Move closer.";
    if (q.faceHeightRatio > QUALITY_RULES.maxFaceHeightRatio) return "Move slightly farther.";
    if (!q.centered) return "Center your face inside the green contour.";
    return "Good image.";
  }

  private getImageDataForBox(sourceCanvas: HTMLCanvasElement, box: any): ImageData {
    const ctx = sourceCanvas.getContext("2d");
    if (!ctx) throw new Error("Could not create canvas context.");
    const x = Math.max(0, Math.floor(box.x));
    const y = Math.max(0, Math.floor(box.y));
    const width = Math.max(1, Math.min(sourceCanvas.width - x, Math.floor(box.width)));
    const height = Math.max(1, Math.min(sourceCanvas.height - y, Math.floor(box.height)));
    return ctx.getImageData(x, y, width, height);
  }

  private brightnessScore(imageData: ImageData): number {
    const data = imageData.data;
    let total = 0;
    const pixels = data.length / 4;
    for (let i = 0; i < data.length; i += 4) total += 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    return total / pixels;
  }

  private blurScore(imageData: ImageData): number {
    const { width, height, data } = imageData;
    if (width < 3 || height < 3) return 0;
    const gray = new Float32Array(width * height);
    for (let i = 0, j = 0; i < data.length; i += 4, j++) gray[j] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];

    let sum = 0;
    let sumSq = 0;
    let count = 0;
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const idx = y * width + x;
        const lap = 4 * gray[idx] - gray[idx - 1] - gray[idx + 1] - gray[idx - width] - gray[idx + width];
        sum += lap;
        sumSq += lap * lap;
        count++;
      }
    }
    const mean = sum / count;
    return sumSq / count - mean * mean;
  }

  private isFaceCentered(box: any, frameWidth: number, frameHeight: number): boolean {
    const centerX = box.x + box.width / 2;
    const centerY = box.y + box.height / 2;
    return centerX >= frameWidth * 0.24 && centerX <= frameWidth * 0.76 && centerY >= frameHeight * 0.15 && centerY <= frameHeight * 0.85;
  }

  private estimatePose(result: any): { yaw: number; pitch: number } {
    const landmarks = result.landmarks;
    const leftEye = averagePoint(landmarks.getLeftEye());
    const rightEye = averagePoint(landmarks.getRightEye());
    const nose = landmarks.getNose()[3] || landmarks.getNose()[0];
    const mouth = averagePoint(landmarks.getMouth());
    const eyeMid = { x: (leftEye.x + rightEye.x) / 2, y: (leftEye.y + rightEye.y) / 2 };
    const eyeDistance = Math.max(1, Math.hypot(rightEye.x - leftEye.x, rightEye.y - leftEye.y));
    const vertical = Math.max(1, mouth.y - eyeMid.y);
    return {
      yaw: (nose.x - eyeMid.x) / eyeDistance,
      pitch: (nose.y - eyeMid.y) / vertical - 0.48,
    };
  }
}
