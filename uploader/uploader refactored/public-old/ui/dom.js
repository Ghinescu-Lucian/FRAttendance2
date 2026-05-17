export class EnrollmentView {
    constructor() {
        this.video = this.required("video");
        this.canvas = this.required("canvas");
        this.startCameraBtn = this.required("startCameraBtn");
        this.captureBtn = this.required("captureBtn");
        this.stopCameraBtn = this.required("stopCameraBtn");
        this.studentIdInput = this.required("studentId");
        this.personNameInput = this.required("personName");
        this.statusEl = this.required("status");
        this.stepEl = this.required("step");
        this.previewsEl = this.required("previews");
    }
    setStatus(message) {
        this.statusEl.textContent = message;
    }
    setStep(message) {
        this.stepEl.textContent = message;
    }
    setCameraRunning(isRunning) {
        this.startCameraBtn.disabled = isRunning;
        this.captureBtn.disabled = !isRunning;
        this.stopCameraBtn.disabled = !isRunning;
    }
    setCaptureEnabled(enabled) {
        this.captureBtn.disabled = !enabled;
    }
    clearPreviews() {
        this.previewsEl.innerHTML = "";
    }
    addPreview(blob, label) {
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
    required(id) {
        const element = document.getElementById(id);
        if (!element)
            throw new Error(`Missing required DOM element: #${id}`);
        return element;
    }
}
