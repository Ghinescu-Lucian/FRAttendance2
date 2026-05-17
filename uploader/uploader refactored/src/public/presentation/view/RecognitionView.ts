import { RecognitionResult } from "../../domain/types";

type FaceBox = {
  x: number;
  y: number;
  width: number;
  height: number;
};

export class RecognitionView {
  readonly video = this.required<HTMLVideoElement>("recognitionVideo");
  readonly canvas = this.required<HTMLCanvasElement>("recognitionCanvas");
  readonly overlayCanvas = this.required<HTMLCanvasElement>("recognitionOverlayCanvas");
  readonly startBtn = this.required<HTMLButtonElement>("startRecognitionCameraBtn");
  readonly stopBtn = this.required<HTMLButtonElement>("stopRecognitionCameraBtn");
  readonly refreshBtn = this.required<HTMLButtonElement>("refreshKnownFacesBtn");
  readonly statusEl = this.required<HTMLParagraphElement>("recognitionStatus");
  readonly resultEl = this.required<HTMLParagraphElement>("recognitionResult");
  readonly knownFacesEl = this.required<HTMLParagraphElement>("knownFacesCount");

  setStatus(message: string): void {
    this.statusEl.textContent = message;
  }

  setKnownFacesCount(count: number): void {
    this.knownFacesEl.textContent = `Known embeddings loaded: ${count}`;
  }

  setCameraRunning(isRunning: boolean): void {
    this.startBtn.disabled = isRunning;
    this.stopBtn.disabled = !isRunning;
  }

  setResult(result: RecognitionResult | null): void {
    if (!result) {
      this.resultEl.textContent = "No recognition result yet.";
      this.resultEl.className = "recognition-result neutral";
      return;
    }

    if (!result.best) {
      this.resultEl.textContent = "No enrolled embeddings available.";
      this.resultEl.className = "recognition-result neutral";
      return;
    }

    if (result.matched) {
      this.resultEl.textContent =
        `MATCH: ${result.best.name} (${result.best.studentId}) — distance ${result.best.distance.toFixed(3)}, similarity ${result.best.similarity.toFixed(3)}`;
      this.resultEl.className = "recognition-result match";
      return;
    }

    this.resultEl.textContent =
      `UNKNOWN — closest: ${result.best.name} (${result.best.studentId}), distance ${result.best.distance.toFixed(3)}, similarity ${result.best.similarity.toFixed(3)}`;
    this.resultEl.className = "recognition-result unknown";
  }

  clearOverlay(): void {
    const ctx = this.prepareOverlayContext();
    if (!ctx) return;
    ctx.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);
  }

  drawOverlay(result: any | null, recognition: RecognitionResult | null): void {
    const ctx = this.prepareOverlayContext();
    if (!ctx) return;

    const width = this.overlayCanvas.width;
    const height = this.overlayCanvas.height;

    ctx.clearRect(0, 0, width, height);

    const box = result?.detection?.box;
    if (!box) {
      this.drawNoFaceGuide(ctx, width, height);
      return;
    }

    const matched = Boolean(recognition?.matched);
    this.drawFaceMarker(ctx, box, matched);
  }

  private drawNoFaceGuide(ctx: CanvasRenderingContext2D, width: number, height: number): void {
    const cx = width / 2;
    const cy = height * 0.5;
    const radius = Math.min(width, height) * 0.12;

    ctx.save();
    ctx.lineWidth = 5;
    ctx.strokeStyle = "rgba(250, 204, 21, 0.95)";
    ctx.setLineDash([12, 10]);
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.stroke();

    ctx.setLineDash([]);
    this.drawLabel(ctx, cx, cy + radius + 36, "PLACE FACE IN CAMERA");
    ctx.restore();
  }

  private drawFaceMarker(ctx: CanvasRenderingContext2D, box: FaceBox, matched: boolean): void {
    const cx = box.x + box.width / 2;
    const cy = box.y + box.height / 2;
    const rx = box.width * 0.55;
    const ry = box.height * 0.62;

    ctx.save();

    ctx.fillStyle = matched ? "rgba(34, 197, 94, 0.12)" : "rgba(248, 113, 113, 0.12)";
    ctx.beginPath();
    ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
    ctx.fill();

    ctx.lineWidth = 8;
    ctx.strokeStyle = "rgba(2, 6, 23, 0.76)";
    ctx.beginPath();
    ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
    ctx.stroke();

    ctx.lineWidth = 4;
    ctx.strokeStyle = matched ? "rgba(34, 197, 94, 0.98)" : "rgba(248, 113, 113, 0.98)";
    ctx.beginPath();
    ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
    ctx.stroke();

    this.drawLabel(ctx, cx, Math.max(28, cy - ry - 20), matched ? "KNOWN" : "UNKNOWN");

    ctx.restore();
  }

  private drawLabel(ctx: CanvasRenderingContext2D, x: number, y: number, text: string): void {
    ctx.save();

    ctx.font = "900 17px Inter, Arial, sans-serif";
    const paddingX = 15;
    const boxHeight = 34;
    const boxWidth = ctx.measureText(text).width + paddingX * 2;
    const boxX = x - boxWidth / 2;
    const boxY = y - boxHeight / 2;

    ctx.fillStyle = "rgba(2, 6, 23, 0.84)";
    this.roundRectPath(ctx, boxX, boxY, boxWidth, boxHeight, 17);
    ctx.fill();

    ctx.fillStyle = "rgba(255,255,255,0.98)";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, x, y + 1);

    ctx.restore();
  }

  private prepareOverlayContext(): CanvasRenderingContext2D | null {
    const width = this.video.videoWidth || this.video.clientWidth || 640;
    const height = this.video.videoHeight || this.video.clientHeight || 480;

    if (this.overlayCanvas.width !== width) this.overlayCanvas.width = width;
    if (this.overlayCanvas.height !== height) this.overlayCanvas.height = height;

    return this.overlayCanvas.getContext("2d");
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
