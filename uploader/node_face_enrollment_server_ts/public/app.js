"use strict";
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const startCameraBtn = document.getElementById("startCameraBtn");
const captureBtn = document.getElementById("captureBtn");
const stopCameraBtn = document.getElementById("stopCameraBtn");
const studentIdInput = document.getElementById("studentId");
const personNameInput = document.getElementById("personName");
const photoCountInput = document.getElementById("photoCount");
const statusEl = document.getElementById("status");
const countdownEl = document.getElementById("countdown");
const previewsEl = document.getElementById("previews");
let stream = null;
function setStatus(message) {
    statusEl.textContent = message;
}
function sleep(ms) {
    return new Promise((resolve) => window.setTimeout(resolve, ms));
}
async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: "user",
                width: { ideal: 1280 },
                height: { ideal: 720 },
            },
            audio: false,
        });
        video.srcObject = stream;
        startCameraBtn.disabled = true;
        captureBtn.disabled = false;
        stopCameraBtn.disabled = false;
        setStatus("Camera started. Put your face inside the green contour.");
    }
    catch (err) {
        const error = err;
        console.error(error);
        setStatus(`Camera error: ${error.name}: ${error.message}`);
    }
}
function stopCamera() {
    if (stream) {
        for (const track of stream.getTracks()) {
            track.stop();
        }
    }
    stream = null;
    video.srcObject = null;
    startCameraBtn.disabled = false;
    captureBtn.disabled = true;
    stopCameraBtn.disabled = true;
    setStatus("Camera stopped.");
}
function validateInputs() {
    const studentId = studentIdInput.value.trim();
    const personName = personNameInput.value.trim();
    if (!studentId) {
        throw new Error("Student ID is required.");
    }
    if (!personName) {
        throw new Error("Person name is required.");
    }
    return { studentId, personName };
}
async function captureImageBlob() {
    if (!video.videoWidth || !video.videoHeight) {
        throw new Error("Video is not ready yet.");
    }
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
        throw new Error("Could not create canvas context.");
    }
    // The preview is mirrored with CSS.
    // The saved image is non-mirrored because this draws the raw camera frame.
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return new Promise((resolve, reject) => {
        canvas.toBlob((blob) => {
            if (!blob) {
                reject(new Error("Could not create image blob."));
                return;
            }
            resolve(blob);
        }, "image/jpeg", 0.92);
    });
}
async function uploadImage(blob, frameIndex) {
    const { studentId, personName } = validateInputs();
    const formData = new FormData();
    formData.append("studentId", studentId);
    formData.append("name", personName);
    formData.append("frameIndex", String(frameIndex));
    formData.append("image", blob, `frame_${frameIndex}.jpg`);
    const response = await fetch("/api/upload-face", {
        method: "POST",
        body: formData,
    });
    const data = (await response.json().catch(() => null));
    if (!response.ok || !data?.ok) {
        throw new Error(data?.error || `Upload failed with status ${response.status}`);
    }
    return data;
}
async function completeEnrollment() {
    const { studentId, personName } = validateInputs();
    const response = await fetch("/api/complete-enrollment", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            studentId,
            name: personName,
        }),
    });
    if (!response.ok) {
        throw new Error(`Complete enrollment failed with status ${response.status}`);
    }
    return (await response.json());
}
function addPreview(blob) {
    const img = document.createElement("img");
    img.src = URL.createObjectURL(blob);
    previewsEl.prepend(img);
}
async function captureSequence() {
    captureBtn.disabled = true;
    previewsEl.innerHTML = "";
    try {
        validateInputs();
        const requestedCount = Number(photoCountInput.value || 5);
        const count = Math.max(1, Math.min(10, requestedCount));
        for (let i = 1; i <= count; i += 1) {
            countdownEl.textContent = String(i);
            setStatus(`Capturing photo ${i}/${count}. Keep your face inside the green contour.`);
            await sleep(650);
            const blob = await captureImageBlob();
            addPreview(blob);
            const result = await uploadImage(blob, i);
            console.log("Uploaded:", result);
            await sleep(250);
        }
        countdownEl.textContent = "";
        const summary = await completeEnrollment();
        setStatus(`Done. Server sees ${summary.savedImages} image(s) for ${summary.name}. Saved to: ${summary.uploadDir}`);
    }
    catch (err) {
        const error = err;
        console.error(error);
        countdownEl.textContent = "";
        setStatus(`Error: ${error.message}`);
    }
    finally {
        captureBtn.disabled = false;
    }
}
startCameraBtn.addEventListener("click", () => void startCamera());
stopCameraBtn.addEventListener("click", stopCamera);
captureBtn.addEventListener("click", () => void captureSequence());
