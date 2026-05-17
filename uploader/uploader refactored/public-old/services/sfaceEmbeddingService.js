import { ORT_WASM_PATH, SFACE_MODEL_URL, SFACE_REFERENCE_POINTS, SFACE_SIZE } from "../config/appConfig.js";
import { averagePoint, l2Normalize } from "../utils/math.js";
export class SFaceEmbeddingService {
    constructor() {
        this.session = null;
    }
    async load() {
        if (this.session)
            return;
        if (typeof ort === "undefined")
            throw new Error("onnxruntime-web was not loaded. Add ort.min.js in index.html.");
        ort.env.wasm.wasmPaths = ORT_WASM_PATH;
        this.session = await ort.InferenceSession.create(SFACE_MODEL_URL, { executionProviders: ["wasm"] });
    }
    async extract(frameCanvas, result) {
        if (!this.session)
            throw new Error("SFace ONNX session is not loaded.");
        const alignedCanvas = this.alignFace(frameCanvas, result);
        const inputTensor = this.canvasToInputTensor(alignedCanvas);
        const inputName = this.session.inputNames[0];
        const outputName = this.session.outputNames[0];
        const outputs = await this.session.run({ [inputName]: inputTensor });
        const raw = Array.from(outputs[outputName].data);
        if (raw.length !== 128)
            throw new Error(`SFace descriptor length is ${raw.length}, expected 128.`);
        return l2Normalize(raw);
    }
    alignFace(sourceCanvas, result) {
        const srcPoints = this.getSourcePoints(result);
        const [a, b, c, d, e, f] = this.getSimilarityTransform(srcPoints, [...SFACE_REFERENCE_POINTS]);
        const alignedCanvas = document.createElement("canvas");
        alignedCanvas.width = SFACE_SIZE;
        alignedCanvas.height = SFACE_SIZE;
        const ctx = alignedCanvas.getContext("2d");
        if (!ctx)
            throw new Error("Could not create aligned face canvas context.");
        ctx.setTransform(a, b, c, d, e, f);
        ctx.drawImage(sourceCanvas, 0, 0);
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        return alignedCanvas;
    }
    getSourcePoints(result) {
        const landmarks = result.landmarks;
        const mouth = landmarks.getMouth();
        return [
            averagePoint(landmarks.getLeftEye()),
            averagePoint(landmarks.getRightEye()),
            landmarks.getNose()[3] || landmarks.getNose()[0],
            mouth[0],
            mouth[6],
        ];
    }
    getSimilarityTransform(src, dst) {
        if (src.length !== 5 || dst.length !== 5)
            throw new Error("SFace alignment needs exactly 5 landmarks.");
        const srcMean = averagePoint(src);
        const dstMean = averagePoint(dst);
        let den = 0;
        let aNum = 0;
        let bNum = 0;
        for (let i = 0; i < src.length; i++) {
            const sx = src[i].x - srcMean.x;
            const sy = src[i].y - srcMean.y;
            const dx = dst[i].x - dstMean.x;
            const dy = dst[i].y - dstMean.y;
            den += sx * sx + sy * sy;
            aNum += sx * dx + sy * dy;
            bNum += sx * dy - sy * dx;
        }
        if (den <= 1e-6)
            throw new Error("Invalid face landmarks for SFace alignment.");
        const a = aNum / den;
        const b = bNum / den;
        const c = -b;
        const d = a;
        const e = dstMean.x - a * srcMean.x - c * srcMean.y;
        const f = dstMean.y - b * srcMean.x - d * srcMean.y;
        return [a, b, c, d, e, f];
    }
    canvasToInputTensor(alignedCanvas) {
        const ctx = alignedCanvas.getContext("2d");
        if (!ctx)
            throw new Error("Could not create aligned face canvas context.");
        const imageData = ctx.getImageData(0, 0, SFACE_SIZE, SFACE_SIZE).data;
        const chw = new Float32Array(1 * 3 * SFACE_SIZE * SFACE_SIZE);
        const planeSize = SFACE_SIZE * SFACE_SIZE;
        for (let y = 0; y < SFACE_SIZE; y++) {
            for (let x = 0; x < SFACE_SIZE; x++) {
                const srcIdx = (y * SFACE_SIZE + x) * 4;
                const dstIdx = y * SFACE_SIZE + x;
                chw[dstIdx] = imageData[srcIdx];
                chw[planeSize + dstIdx] = imageData[srcIdx + 1];
                chw[2 * planeSize + dstIdx] = imageData[srcIdx + 2];
            }
        }
        return new ort.Tensor("float32", chw, [1, 3, SFACE_SIZE, SFACE_SIZE]);
    }
}
