export type CameraDevice = {
  deviceId: string;
  label: string;
};

export class CameraService {
  private stream: MediaStream | null = null;

  constructor(private readonly video: HTMLVideoElement) {}

  async listCameras(): Promise<CameraDevice[]> {
    if (!navigator.mediaDevices?.enumerateDevices) return [];

    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices
      .filter((device) => device.kind === "videoinput")
      .map((device, index) => ({
        deviceId: device.deviceId,
        label: device.label || `Camera ${index + 1}`,
      }));
  }

  async start(deviceId?: string): Promise<void> {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("Camera API is not available. Open Moodle through HTTPS or localhost, then allow camera permission.");
    }

    // Release the previous camera before requesting another one.
    // This is important on Windows/laptops because some browsers keep the
    // built-in camera locked and then silently fall back to it.
    this.stop();

    const constraints = this.buildConstraints(deviceId);

    let lastError: unknown = null;
    for (const constraint of constraints) {
      try {
        this.stream = await navigator.mediaDevices.getUserMedia(constraint);
        break;
      } catch (err) {
        lastError = err;
      }
    }

    if (!this.stream) {
      const error = lastError as Error | null;
      if (deviceId) {
        throw new Error(error?.message || "Could not start the selected camera. Press Refresh cameras, select it again, then start.");
      }
      throw new Error(error?.message || "Could not start any available camera.");
    }

    this.video.autoplay = true;
    this.video.muted = true;
    this.video.playsInline = true;
    this.video.srcObject = this.stream;

    await this.waitForMetadata();
    await this.video.play();
    await this.waitForVideoSize();
  }

  stop(): void {
    if (this.stream) for (const track of this.stream.getTracks()) track.stop();
    this.stream = null;
    this.video.pause();
    this.video.srcObject = null;
  }

  getActiveDeviceId(): string | null {
    const track = this.stream?.getVideoTracks()[0];
    const settings = track?.getSettings?.();
    return settings?.deviceId || null;
  }

  captureFrame(targetCanvas: HTMLCanvasElement): HTMLCanvasElement {
    if (!this.video.videoWidth || !this.video.videoHeight) throw new Error("Video is not ready yet.");
    targetCanvas.width = this.video.videoWidth;
    targetCanvas.height = this.video.videoHeight;
    const ctx = targetCanvas.getContext("2d");
    if (!ctx) throw new Error("Could not create canvas context.");
    ctx.drawImage(this.video, 0, 0, targetCanvas.width, targetCanvas.height);
    return targetCanvas;
  }

  private buildConstraints(deviceId?: string): MediaStreamConstraints[] {
    if (deviceId) {
      // Do not fall back to { ideal: deviceId } here. If the selected camera
      // cannot be opened, showing an error is better than silently opening the
      // laptop built-in camera and making it look like selection is ignored.
      return [
        {
          video: {
            deviceId: { exact: deviceId },
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: false,
        },
      ];
    }

    return [
      {
        video: {
          facingMode: { ideal: "user" },
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      },
      {
        video: true,
        audio: false,
      },
    ];
  }

  private waitForMetadata(): Promise<void> {
    if (this.video.readyState >= HTMLMediaElement.HAVE_METADATA) return Promise.resolve();

    return new Promise((resolve, reject) => {
      const timeout = window.setTimeout(() => {
        cleanup();
        reject(new Error("Camera stream started, but video metadata was not loaded."));
      }, 8000);

      const onLoadedMetadata = () => {
        cleanup();
        resolve();
      };

      const onError = () => {
        cleanup();
        reject(new Error("The browser could not attach the camera stream to the video element."));
      };

      const cleanup = () => {
        window.clearTimeout(timeout);
        this.video.removeEventListener("loadedmetadata", onLoadedMetadata);
        this.video.removeEventListener("error", onError);
      };

      this.video.addEventListener("loadedmetadata", onLoadedMetadata, { once: true });
      this.video.addEventListener("error", onError, { once: true });
    });
  }

  private async waitForVideoSize(): Promise<void> {
    const timeoutAt = Date.now() + 8000;
    while (Date.now() < timeoutAt) {
      if (this.video.videoWidth > 0 && this.video.videoHeight > 0) return;
      await new Promise((resolve) => window.setTimeout(resolve, 100));
    }
    throw new Error("Camera opened, but the browser did not provide video frames.");
  }
}
