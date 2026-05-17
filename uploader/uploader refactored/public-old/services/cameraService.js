export class CameraService {
    constructor(video) {
        this.video = video;
        this.stream = null;
    }
    async start() {
        if (!navigator.mediaDevices?.getUserMedia) {
            throw new Error("Camera API is not available. Use HTTPS or localhost.");
        }
        this.stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
            audio: false,
        });
        this.video.srcObject = this.stream;
    }
    stop() {
        if (this.stream)
            for (const track of this.stream.getTracks())
                track.stop();
        this.stream = null;
        this.video.srcObject = null;
    }
    captureFrame(targetCanvas) {
        if (!this.video.videoWidth || !this.video.videoHeight)
            throw new Error("Video is not ready yet.");
        targetCanvas.width = this.video.videoWidth;
        targetCanvas.height = this.video.videoHeight;
        const ctx = targetCanvas.getContext("2d");
        if (!ctx)
            throw new Error("Could not create canvas context.");
        ctx.drawImage(this.video, 0, 0, targetCanvas.width, targetCanvas.height);
        return targetCanvas;
    }
}
