import { CaptureStep } from "../../domain/types";

type FaceBox = {
  x: number;
  y: number;
  width: number;
  height: number;
};

export class EnrollmentView {
  readonly video = this.required<HTMLVideoElement>("video");
  readonly canvas = this.required<HTMLCanvasElement>("canvas");
  readonly overlayCanvas = this.required<HTMLCanvasElement>("overlayCanvas");
  readonly startCameraBtn = this.required<HTMLButtonElement>("startCameraBtn");
  readonly captureBtn = this.required<HTMLButtonElement>("captureBtn");
  readonly stopCameraBtn = this.required<HTMLButtonElement>("stopCameraBtn");
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

    this.drawTargetFrame(ctx, width, height);

    if (this.activeStep) {
      this.drawActionIndicator(ctx, width, height, this.activeStep.pose, time);
    }

    const box = this.activeResult?.detection?.box;
    if (box) this.drawDetectedFace(ctx, box);
  }

  private prepareOverlayContext(): CanvasRenderingContext2D | null {
    const width = this.video.videoWidth || this.video.clientWidth || 640;
    const height = this.video.videoHeight || this.video.clientHeight || 480;

    if (this.overlayCanvas.width !== width) this.overlayCanvas.width = width;
    if (this.overlayCanvas.height !== height) this.overlayCanvas.height = height;

    return this.overlayCanvas.getContext("2d");
  }

  private drawTargetFrame(ctx: CanvasRenderingContext2D, width: number, height: number): void {
    const guideWidth = width * 0.42;
    const guideHeight = height * 0.62;
    const x = (width - guideWidth) / 2;
    const y = (height - guideHeight) / 2 - height * 0.04;

    ctx.save();

    ctx.lineWidth = 8;
    ctx.strokeStyle = "rgba(2, 6, 23, 0.72)";
    ctx.setLineDash([16, 11]);
    this.roundRectPath(ctx, x, y, guideWidth, guideHeight, 28);
    ctx.stroke();

    ctx.lineWidth = 4;
    ctx.strokeStyle = "rgba(34, 197, 94, 0.98)";
    ctx.setLineDash([16, 11]);
    this.roundRectPath(ctx, x, y, guideWidth, guideHeight, 28);
    ctx.stroke();

    ctx.setLineDash([]);
    ctx.lineWidth = 2;
    ctx.strokeStyle = "rgba(255, 255, 255, 0.65)";
    this.roundRectPath(ctx, x + 8, y + 8, guideWidth - 16, guideHeight - 16, 22);
    ctx.stroke();

    ctx.restore();
  }

  private drawDetectedFace(ctx: CanvasRenderingContext2D, box: FaceBox): void {
    ctx.save();

    ctx.fillStyle = "rgba(56, 189, 248, 0.14)";
    this.roundRectPath(ctx, box.x, box.y, box.width, box.height, 18);
    ctx.fill();

    ctx.lineWidth = 8;
    ctx.strokeStyle = "rgba(2, 6, 23, 0.75)";
    this.roundRectPath(ctx, box.x, box.y, box.width, box.height, 18);
    ctx.stroke();

    ctx.lineWidth = 4;
    ctx.strokeStyle = "rgba(14, 165, 233, 0.98)";
    this.roundRectPath(ctx, box.x, box.y, box.width, box.height, 18);
    ctx.stroke();

    ctx.font = "800 16px Inter, Arial, sans-serif";
    ctx.fillStyle = "rgba(255,255,255,0.98)";
    ctx.shadowColor = "rgba(0,0,0,0.85)";
    ctx.shadowBlur = 5;
    ctx.fillText("FACE", box.x + 10, Math.max(24, box.y - 10));

    ctx.restore();
  }

  private drawActionIndicator(
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    pose: string,
    time: number,
  ): void {
    const centerX = width / 2;
    const centerY = height * 0.82;
    const distance = Math.min(width, height) * 0.24;
    const t = (Math.sin(time * Math.PI * 1.8) + 1) / 2;

    let x1 = centerX;
    let y1 = centerY;
    let x2 = centerX;
    let y2 = centerY;
    let label = "LOOK STRAIGHT";
    let angle = 0;

    if (pose === "left") {
      x1 = centerX + distance;
      x2 = centerX - distance;
      label = "MOVE / TURN LEFT";
      angle = -Math.PI / 2;
    } else if (pose === "right") {
      x1 = centerX - distance;
      x2 = centerX + distance;
      label = "MOVE / TURN RIGHT";
      angle = Math.PI / 2;
    } else if (pose === "down") {
      y1 = centerY - distance * 0.72;
      y2 = centerY + distance * 0.72;
      label = "TILT DOWN";
      angle = Math.PI;
    } else {
      y1 = centerY - 18;
      y2 = centerY + 18;
      label = "LOOK STRAIGHT";
      angle = 0;
    }

    const handX = x1 + (x2 - x1) * t;
    const handY = y1 + (y2 - y1) * t;

    ctx.save();

    this.drawDirectionArrow(ctx, x1, y1, x2, y2);
    this.drawWaterWaves(ctx, handX, handY, time);
    this.drawPointerHand(ctx, handX, handY, angle, 1 + 0.08 * Math.sin(time * Math.PI * 3.6));
    this.drawDirectionLabel(ctx, centerX, Math.min(height - 28, centerY + 74), label);

    ctx.restore();
  }

  private drawDirectionArrow(
    ctx: CanvasRenderingContext2D,
    x1: number,
    y1: number,
    x2: number,
    y2: number,
  ): void {
    const angle = Math.atan2(y2 - y1, x2 - x1);
    const head = 28;

    ctx.save();

    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    ctx.lineWidth = 14;
    ctx.strokeStyle = "rgba(2, 6, 23, 0.76)";
    this.drawArrowPath(ctx, x1, y1, x2, y2, angle, head);

    ctx.lineWidth = 8;
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

  private drawWaterWaves(ctx: CanvasRenderingContext2D, x: number, y: number, time: number): void {
    ctx.save();
    ctx.lineWidth = 4;

    for (let i = 0; i < 3; i++) {
      const phase = (time * 1.7 + i * 0.32) % 1;
      const radiusX = 34 + phase * 58;
      const radiusY = 14 + phase * 28;
      const alpha = 0.72 * (1 - phase);

      ctx.strokeStyle = `rgba(125, 211, 252, ${alpha})`;
      ctx.beginPath();
      ctx.ellipse(x, y + 20, radiusX, radiusY, 0, 0, Math.PI * 2);
      ctx.stroke();
    }

    ctx.restore();
  }

  private drawPointerHand(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    angle: number,
    scale: number,
  ): void {
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(angle);
    ctx.scale(scale, scale);

    ctx.shadowColor = "rgba(0,0,0,0.78)";
    ctx.shadowBlur = 12;
    ctx.shadowOffsetY = 5;

    // Dark outline under the whole hand, so it is visible on any camera background.
    ctx.fillStyle = "rgba(2, 6, 23, 0.82)";
    this.roundRectPath(ctx, -18, -74, 36, 86, 18);
    ctx.fill();
    this.roundRectPath(ctx, -42, -6, 84, 54, 24);
    ctx.fill();

    // Palm.
    ctx.shadowBlur = 0;
    ctx.shadowOffsetY = 0;
    ctx.fillStyle = "rgba(254, 215, 170, 0.98)";
    this.roundRectPath(ctx, -34, -2, 68, 52, 22);
    ctx.fill();

    // Index finger.
    this.roundRectPath(ctx, -13, -82, 26, 88, 13);
    ctx.fill();

    // Other fingers.
    this.roundRectPath(ctx, -34, -20, 20, 52, 10);
    ctx.fill();
    this.roundRectPath(ctx, 14, -16, 20, 48, 10);
    ctx.fill();

    // Thumb.
    ctx.save();
    ctx.translate(-32, 12);
    ctx.rotate(-0.72);
    this.roundRectPath(ctx, -8, -8, 24, 48, 12);
    ctx.fill();
    ctx.restore();

    // Fingernail / highlight.
    ctx.fillStyle = "rgba(255, 247, 237, 0.92)";
    this.roundRectPath(ctx, -8, -76, 16, 13, 7);
    ctx.fill();

    // Palm contour.
    ctx.strokeStyle = "rgba(124, 45, 18, 0.42)";
    ctx.lineWidth = 2;
    this.roundRectPath(ctx, -34, -2, 68, 52, 22);
    ctx.stroke();
    this.roundRectPath(ctx, -13, -82, 26, 88, 13);
    ctx.stroke();

    ctx.restore();
  }

  private drawDirectionLabel(ctx: CanvasRenderingContext2D, x: number, y: number, label: string): void {
    const paddingX = 18;
    const boxHeight = 38;

    ctx.save();

    ctx.font = "900 18px Inter, Arial, sans-serif";
    const width = ctx.measureText(label).width + paddingX * 2;
    const boxX = x - width / 2;
    const boxY = y - boxHeight / 2;

    ctx.fillStyle = "rgba(2, 6, 23, 0.84)";
    this.roundRectPath(ctx, boxX, boxY, width, boxHeight, 19);
    ctx.fill();

    ctx.strokeStyle = "rgba(255,255,255,0.42)";
    ctx.lineWidth = 1;
    this.roundRectPath(ctx, boxX, boxY, width, boxHeight, 19);
    ctx.stroke();

    ctx.fillStyle = "rgba(255,255,255,0.98)";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(label, x, y + 1);

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
