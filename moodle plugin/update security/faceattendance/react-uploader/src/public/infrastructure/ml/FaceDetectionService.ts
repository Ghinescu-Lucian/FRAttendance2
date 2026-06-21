import { FACE_API_MODEL_URL, FACE_DETECTION_OPTIONS } from "../../shared/appConfig";

export class FaceDetectionService {
  private loaded = false;

  async load(): Promise<void> {
    if (this.loaded) return;
    if (typeof faceapi === "undefined") throw new Error("face-api library was not loaded.");

    await Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri(FACE_API_MODEL_URL),
      faceapi.nets.faceLandmark68Net.loadFromUri(FACE_API_MODEL_URL),
    ]);
    this.loaded = true;
  }

  async detect(video: HTMLVideoElement): Promise<any | null> {
    if (!video.videoWidth || !video.videoHeight || !this.loaded) return null;
    return await faceapi.detectSingleFace(video, this.options()).withFaceLandmarks();
  }

  async detectAll(video: HTMLVideoElement): Promise<any[]> {
    if (!video.videoWidth || !video.videoHeight || !this.loaded) return [];
    return await faceapi.detectAllFaces(video, this.options()).withFaceLandmarks();
  }

  async detectAllFromCanvas(canvas: HTMLCanvasElement): Promise<any[]> {
    if (!canvas.width || !canvas.height || !this.loaded) return [];
    return await faceapi.detectAllFaces(canvas, this.options()).withFaceLandmarks();
  }

  private options(): any {
    return new faceapi.TinyFaceDetectorOptions(FACE_DETECTION_OPTIONS);
  }
}
