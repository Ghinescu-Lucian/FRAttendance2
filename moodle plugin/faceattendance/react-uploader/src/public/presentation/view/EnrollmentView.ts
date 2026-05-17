import { CaptureStep } from "../../domain/types";

type FaceBox = {
  x: number;
  y: number;
  width: number;
  height: number;
};

export class EnrollmentView {
  readonly video = this.required<HTMLVideoElement>("video");
  readonly canvas = this.required<HTMLCanvasElement>("canvas"); // hidden capture canvas
  readonly overlayCanvas = this.required<HTMLCanvasElement>("overlayCanvas"); // visible overlay canvas
  readonly startCameraBtn = this.required<HTMLButtonElement>("startCameraBtn");
  readonly captureBtn = this.required<HTMLButtonElement>("captureBtn");
  readonly stopCameraBtn = this.required<HTMLButtonElement>("stopCameraBtn");
  readonly cameraSelect = this.required<HTMLSelectElement>("cameraSelect");
  readonly studentIdInput = this.required<HTMLInputElement>("studentId");
  readonly personNameInput = this.required<HTMLInputElement>("personName");
  readonly statusEl = this.required<HTMLParagraphElement>("status");
  readonly stepEl = this.required<HTMLParagraphElement>("step");
  readonly previewsEl = this.required<HTMLDivElement>("previews");

  private overlayAnimationId: number | null = null;
  private activeStep: CaptureStep | null = null;
  private activeResult: any | null = null;
  private startedAt = 0;

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

  getSelectedCameraId(): string | undefined {
    return this.cameraSelect.value || undefined;
  }

  setCameraOptions(devices: Array<{ deviceId: string; label: string }>, activeDeviceId?: string | null): void {
    const previous = activeDeviceId || this.cameraSelect.value;
    this.cameraSelect.innerHTML = "";

    const defaultOption = document.createElement("option");
    defaultOption.value = "";
    defaultOption.textContent = devices.length ? "Default camera" : "No camera found yet";
    this.cameraSelect.appendChild(defaultOption);

    for (const device of devices) {
      const option = document.createElement("option");
      option.value = device.deviceId;
      option.textContent = device.label;
      this.cameraSelect.appendChild(option);
    }

    if (previous && devices.some((device) => device.deviceId === previous)) {
      this.cameraSelect.value = previous;
    }
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

  startGuidance(step: CaptureStep): void {
    this.activeStep = step;
    this.activeResult = null;
    this.startedAt = performance.now();

    if (this.overlayAnimationId !== null) return;

    const draw = () => {
      this.drawOverlay();
      this.overlayAnimationId = window.requestAnimationFrame(draw);
    };

    this.overlayAnimationId = window.requestAnimationFrame(draw);
  }

  updateFaceDetection(result: any | null): void {
    this.activeResult = result;
  }

  clearOverlay(): void {
    if (this.overlayAnimationId !== null) {
      window.cancelAnimationFrame(this.overlayAnimationId);
      this.overlayAnimationId = null;
    }

    this.activeStep = null;
    this.activeResult = null;

    const ctx = this.prepareOverlayContext();
    if (!ctx) return;
    ctx.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);
  }

  private drawOverlay(): void {
    const ctx = this.prepareOverlayContext();
    if (!ctx) return;

    const width = this.overlayCanvas.width;
    const height = this.overlayCanvas.height;
    const time = (performance.now() - this.startedAt) / 1000;

    ctx.clearRect(0, 0, width, height);

    if (this.activeStep) {
      this.drawDirectionGesture(ctx, width, height, this.activeStep.pose, time);
    }

    const box = this.activeResult?.detection?.box;
    if (box) {
      this.drawDetectedFaceOval(ctx, box);
    }
  }

  private prepareOverlayContext(): CanvasRenderingContext2D | null {
    const width = this.video.videoWidth || this.video.clientWidth || 640;
    const height = this.video.videoHeight || this.video.clientHeight || 480;

    if (this.overlayCanvas.width !== width) this.overlayCanvas.width = width;
    if (this.overlayCanvas.height !== height) this.overlayCanvas.height = height;

    return this.overlayCanvas.getContext("2d");
  }

  private drawFaceGuide(ctx: CanvasRenderingContext2D, width: number, height: number): void {
    const size = Math.min(width, height);
    const cx = width / 2;
    const cy = height * 0.46;
    const faceWidth = size * 0.34;
    const faceHeight = size * 0.46;

    ctx.save();

    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    // Outer dark contour, so it is readable on any camera background.
    ctx.lineWidth = 8;
    ctx.strokeStyle = "rgba(2, 6, 23, 0.78)";
    this.faceContourPath(ctx, cx, cy, faceWidth, faceHeight);
    ctx.stroke();

    // Main face-shaped guide.
    ctx.setLineDash([13, 10]);
    ctx.lineWidth = 4;
    ctx.strokeStyle = "rgba(34, 197, 94, 0.98)";
    this.faceContourPath(ctx, cx, cy, faceWidth, faceHeight);
    ctx.stroke();

    // Chin / shoulders hint.
    ctx.setLineDash([]);
    ctx.lineWidth = 3;
    ctx.strokeStyle = "rgba(34, 197, 94, 0.70)";
    ctx.beginPath();
    ctx.moveTo(cx - faceWidth * 0.50, cy + faceHeight * 0.48);
    ctx.quadraticCurveTo(cx, cy + faceHeight * 0.67, cx + faceWidth * 0.50, cy + faceHeight * 0.48);
    ctx.stroke();

    ctx.restore();
  }

  private faceContourPath(
    ctx: CanvasRenderingContext2D,
    cx: number,
    cy: number,
    faceWidth: number,
    faceHeight: number,
  ): void {
    const top = cy - faceHeight * 0.52;
    const bottom = cy + faceHeight * 0.54;
    const left = cx - faceWidth * 0.50;
    const right = cx + faceWidth * 0.50;

    ctx.beginPath();
    ctx.moveTo(cx, top);

    // Left forehead and cheek.
    ctx.bezierCurveTo(
      cx - faceWidth * 0.34,
      top + faceHeight * 0.02,
      left,
      cy - faceHeight * 0.20,
      left + faceWidth * 0.05,
      cy + faceHeight * 0.12,
    );

    // Left jaw into chin.
    ctx.bezierCurveTo(
      left + faceWidth * 0.08,
      cy + faceHeight * 0.36,
      cx - faceWidth * 0.18,
      bottom,
      cx,
      bottom,
    );

    // Right jaw.
    ctx.bezierCurveTo(
      cx + faceWidth * 0.18,
      bottom,
      right - faceWidth * 0.08,
      cy + faceHeight * 0.36,
      right - faceWidth * 0.05,
      cy + faceHeight * 0.12,
    );

    // Right cheek and forehead.
    ctx.bezierCurveTo(
      right,
      cy - faceHeight * 0.20,
      cx + faceWidth * 0.34,
      top + faceHeight * 0.02,
      cx,
      top,
    );

    ctx.closePath();
  }

  private drawDetectedFaceOval(ctx: CanvasRenderingContext2D, box: FaceBox): void {
    const cx = box.x + box.width / 2;
    const cy = box.y + box.height / 2;
    const rx = box.width * 0.54;
    const ry = box.height * 0.62;

    ctx.save();

    ctx.fillStyle = "rgba(56, 189, 248, 0.10)";
    ctx.beginPath();
    ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
    ctx.fill();

    ctx.lineWidth = 7;
    ctx.strokeStyle = "rgba(2, 6, 23, 0.72)";
    ctx.beginPath();
    ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
    ctx.stroke();

    ctx.lineWidth = 3;
    ctx.strokeStyle = "rgba(14, 165, 233, 0.98)";
    ctx.beginPath();
    ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
    ctx.stroke();

    ctx.font = "800 14px Inter, Arial, sans-serif";
    ctx.fillStyle = "rgba(255,255,255,0.98)";
    ctx.shadowColor = "rgba(0,0,0,0.85)";
    ctx.shadowBlur = 5;
    ctx.fillText("FACE", cx - 16, Math.max(22, cy - ry - 8));

    ctx.restore();
  }

  private drawDirectionGesture(
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    pose: string,
    time: number,
  ): void {
    const centerX = width / 2;
    const baseY = height * 0.70;
    const distance = Math.min(width, height) * 0.22;
    const progress = (Math.sin(time * Math.PI * 1.65) + 1) / 2;

    let startX = centerX;
    let startY = baseY;
    let endX = centerX;
    let endY = baseY;
    let label = "LOOK STRAIGHT";
    if (pose === "left") {
      startX = centerX + distance;
      endX = centerX - distance;
      label = "TURN LEFT";
    } else if (pose === "right") {
      startX = centerX - distance;
      endX = centerX + distance;
      label = "TURN RIGHT";
    } else if (pose === "down") {
      startY = baseY - distance * 0.65;
      endY = baseY + distance * 0.65;
      label = "TILT DOWN";
    } else {
      startY = baseY - 16;
      endY = baseY + 16;
      label = "LOOK STRAIGHT";
    }

    const handX = startX + (endX - startX) * progress;
    const handY = startY + (endY - startY) * progress;

    this.drawWaterWaves(ctx, handX, handY + 16, time);
    this.drawArrow(ctx, startX, startY, endX, endY);
    this.drawHeadDirectionIcon(ctx, handX, handY - 10, 0.95 + 0.05 * Math.sin(time * Math.PI * 2.2));
    this.drawGestureLabel(ctx, centerX, Math.min(height - 28, baseY + 58), label);
  }

  private drawWaterWaves(ctx: CanvasRenderingContext2D, x: number, y: number, time: number): void {
    ctx.save();
    ctx.lineWidth = 4;

    for (let i = 0; i < 3; i++) {
      const phase = (time * 1.5 + i * 0.33) % 1;
      const rx = 28 + phase * 56;
      const ry = 10 + phase * 24;
      const alpha = 0.72 * (1 - phase);

      ctx.strokeStyle = `rgba(125, 211, 252, ${alpha})`;
      ctx.beginPath();
      ctx.ellipse(x, y, rx, ry, 0, 0, Math.PI * 2);
      ctx.stroke();
    }

    ctx.restore();
  }

  private drawArrow(ctx: CanvasRenderingContext2D, x1: number, y1: number, x2: number, y2: number): void {
    const angle = Math.atan2(y2 - y1, x2 - x1);
    const head = 28;

    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    ctx.lineWidth = 14;
    ctx.strokeStyle = "rgba(2, 6, 23, 0.76)";
    this.drawArrowPath(ctx, x1, y1, x2, y2, angle, head);

    ctx.lineWidth = 7;
    ctx.strokeStyle = "rgba(250, 204, 21, 0.98)";
    this.drawArrowPath(ctx, x1, y1, x2, y2, angle, head);

    ctx.restore();
  }

  private drawArrowPath(
    ctx: CanvasRenderingContext2D,
    x1: number,
    y1: number,
    x2: number,
    y2: number,
    angle: number,
    head: number,
  ): void {
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - head * Math.cos(angle - Math.PI / 6), y2 - head * Math.sin(angle - Math.PI / 6));
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - head * Math.cos(angle + Math.PI / 6), y2 - head * Math.sin(angle + Math.PI / 6));
    ctx.stroke();
  }

  private drawHeadDirectionIcon(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    scale: number,
  ): void {
    ctx.save();
    ctx.translate(x, y);
    ctx.scale(scale, scale);

    // A neutral professional person/head icon. Direction is shown by the moving arrow,
    // not by fingers or emoji.
    ctx.shadowColor = "rgba(0,0,0,0.78)";
    ctx.shadowBlur = 12;
    ctx.shadowOffsetY = 5;

    // Dark contrast disc.
    ctx.fillStyle = "rgba(2, 6, 23, 0.78)";
    ctx.beginPath();
    ctx.arc(0, -18, 44, 0, Math.PI * 2);
    ctx.fill();

    // Face circle.
    ctx.shadowBlur = 0;
    ctx.shadowOffsetY = 0;
    ctx.fillStyle = "rgba(226, 232, 240, 0.96)";
    ctx.beginPath();
    ctx.arc(0, -30, 23, 0, Math.PI * 2);
    ctx.fill();

    // Neck.
    ctx.fillStyle = "rgba(203, 213, 225, 0.96)";
    this.roundRectPath(ctx, -10, -8, 20, 18, 7);
    ctx.fill();

    // Shoulders.
    ctx.fillStyle = "rgba(148, 163, 184, 0.96)";
    ctx.beginPath();
    ctx.moveTo(-42, 32);
    ctx.quadraticCurveTo(-28, 4, 0, 4);
    ctx.quadraticCurveTo(28, 4, 42, 32);
    ctx.lineTo(42, 42);
    ctx.lineTo(-42, 42);
    ctx.closePath();
    ctx.fill();

    // Simple face direction line.
    ctx.strokeStyle = "rgba(15, 23, 42, 0.60)";
    ctx.lineWidth = 3;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(0, -38);
    ctx.lineTo(0, -20);
    ctx.stroke();

    // Tiny eyes, neutral.
    ctx.fillStyle = "rgba(15, 23, 42, 0.70)";
    ctx.beginPath();
    ctx.arc(-8, -32, 2.2, 0, Math.PI * 2);
    ctx.arc(8, -32, 2.2, 0, Math.PI * 2);
    ctx.fill();

    // Outer white highlight.
    ctx.strokeStyle = "rgba(255,255,255,0.72)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(0, -18, 44, 0, Math.PI * 2);
    ctx.stroke();

    ctx.restore();
  }

  private drawGestureLabel(ctx: CanvasRenderingContext2D, x: number, y: number, text: string): void {
    ctx.save();

    ctx.font = "900 18px Inter, Arial, sans-serif";
    const paddingX = 18;
    const boxHeight = 38;
    const boxWidth = ctx.measureText(text).width + paddingX * 2;
    const boxX = x - boxWidth / 2;
    const boxY = y - boxHeight / 2;

    ctx.fillStyle = "rgba(2, 6, 23, 0.86)";
    this.roundRectPath(ctx, boxX, boxY, boxWidth, boxHeight, 19);
    ctx.fill();

    ctx.strokeStyle = "rgba(255,255,255,0.42)";
    ctx.lineWidth = 1;
    this.roundRectPath(ctx, boxX, boxY, boxWidth, boxHeight, 19);
    ctx.stroke();

    ctx.fillStyle = "rgba(255,255,255,0.98)";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, x, y + 1);

    ctx.restore();
  }

  private roundRectPath(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    width: number,
    height: number,
    radius: number,
  ): void {
    const r = Math.min(radius, width / 2, height / 2);

    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + width - r, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + r);
    ctx.lineTo(x + width, y + height - r);
    ctx.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
    ctx.lineTo(x + r, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
  }

  private required<T extends HTMLElement>(id: string): T {
    const element = document.getElementById(id);
    if (!element) throw new Error(`Missing required DOM element: #${id}`);
    return element as T;
  }
}
