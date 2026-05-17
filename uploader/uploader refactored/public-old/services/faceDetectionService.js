import { FACE_API_MODEL_URL } from "../config/appConfig.js";
export class FaceDetectionService {
    constructor() {
        this.loaded = false;
    }
    async load() {
        if (this.loaded)
            return;
        if (typeof faceapi === "undefined")
            throw new Error("face-api library was not loaded.");
        await Promise.all([
            faceapi.nets.tinyFaceDetector.loadFromUri(FACE_API_MODEL_URL),
            faceapi.nets.faceLandmark68Net.loadFromUri(FACE_API_MODEL_URL),
        ]);
        this.loaded = true;
    }
    async detect(video) {
        if (!video.videoWidth || !video.videoHeight || !this.loaded)
            return null;
        return await faceapi.detectSingleFace(video, this.options()).withFaceLandmarks();
    }
    options() {
        return new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.5 });
    }
}
