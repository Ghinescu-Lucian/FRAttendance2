export class EnrollmentView {
  readonly video = this.required<HTMLVideoElement>("video");
  readonly canvas = this.required<HTMLCanvasElement>("canvas");
  readonly startCameraBtn = this.required<HTMLButtonElement>("startCameraBtn");
  readonly captureBtn = this.required<HTMLButtonElement>("captureBtn");
  readonly stopCameraBtn = this.required<HTMLButtonElement>("stopCameraBtn");
  readonly studentIdInput = this.required<HTMLInputElement>("studentId");
  readonly personNameInput = this.required<HTMLInputElement>("personName");
  readonly statusEl = this.required<HTMLParagraphElement>("status");
  readonly stepEl = this.required<HTMLParagraphElement>("step");
  readonly previewsEl = this.required<HTMLDivElement>("previews");

  setStatus(message: string): void {
    this.statusEl.textContent = message;
  }

  setStep(message: string): void {
    this.stepEl.textContent = message;
  }

  setCameraRunning(isRunning: boolean): void {
    this.startCameraBtn.disabled = isRunning;
    this.captureBtn.disabled = !isRunning;
    this.stopCameraBtn.disabled = !isRunning;
  }

  setCaptureEnabled(enabled: boolean): void {
    this.captureBtn.disabled = !enabled;
  }

  clearPreviews(): void {
    this.previewsEl.innerHTML = "";
  }

  addPreview(blob: Blob, label: string): void {
    const wrapper = document.createElement("div");
    wrapper.className = "preview-item";

    const img = document.createElement("img");
    img.src = URL.createObjectURL(blob);

    const caption = document.createElement("span");
    caption.textContent = label;

    wrapper.appendChild(img);
    wrapper.appendChild(caption);
    this.previewsEl.prepend(wrapper);
  }

  private required<T extends HTMLElement>(id: string): T {
    const element = document.getElementById(id);
    if (!element) throw new Error(`Missing required DOM element: #${id}`);
    return element as T;
  }
}
