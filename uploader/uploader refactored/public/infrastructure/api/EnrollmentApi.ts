import { ENROLLMENT_APP_BUILD, SFACE_SIZE } from "../../shared/appConfig";
import { CapturedEmbedding, SFaceEmbeddingPayload, StudentIdentity } from "../../domain/types";
import { descriptorNorm } from "../../shared/math";

export class EnrollmentApi {
  async uploadImage(identity: StudentIdentity, blob: Blob, frameIndex: number, pose: string): Promise<void> {
    const formData = new FormData();
    formData.append("studentId", identity.studentId);
    formData.append("name", identity.personName);
    formData.append("frameIndex", String(frameIndex));
    formData.append("pose", pose);
    formData.append("image", blob, `${pose}_${frameIndex}.jpg`);

    const response = await fetch("/api/upload-face", { method: "POST", body: formData });
    const data = await response.json().catch(() => null);
    if (!response.ok || !data?.ok) throw new Error(data?.error || `Image upload failed with status ${response.status}`);
  }

  async uploadEmbeddingFile(identity: StudentIdentity, embeddings: CapturedEmbedding[]): Promise<void> {
    const payload = this.buildPayload(identity, embeddings);
    this.assertValidPayload(payload);

    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const formData = new FormData();
    formData.append("studentId", identity.studentId);
    formData.append("name", identity.personName);
    formData.append("model", "opencv-sface-2021dec");
    formData.append("clientBuild", ENROLLMENT_APP_BUILD);
    formData.append("embeddingFile", blob, `${identity.personName}_${identity.studentId}_sface_embeddings.json`);

    const response = await fetch("/api/upload-embedding-file", { method: "POST", body: formData });
    const data = await response.json().catch(() => null);
    if (!response.ok || !data?.ok) throw new Error(data?.error || `Embedding upload failed with status ${response.status}`);
  }

  async completeEnrollment(identity: StudentIdentity): Promise<any> {
    const response = await fetch("/api/complete-enrollment", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ studentId: identity.studentId, name: identity.personName }),
    });
    if (!response.ok) throw new Error(`Complete enrollment failed with status ${response.status}`);
    return await response.json();
  }

  private buildPayload(identity: StudentIdentity, embeddings: CapturedEmbedding[]): SFaceEmbeddingPayload {
    return {
      version: 2,
      studentId: identity.studentId,
      name: identity.personName,
      createdAt: new Date().toISOString(),
      model: {
        family: "opencv",
        detectorForEnrollment: "@vladmandic/face-api tinyFaceDetector + faceLandmark68Net",
        recognizer: "sface",
        recognizerModel: "face_recognition_sface_2021dec.onnx",
        descriptorLength: 128,
        descriptorNormalized: true,
        alignedSize: [SFACE_SIZE, SFACE_SIZE],
        clientBuild: ENROLLMENT_APP_BUILD,
        note: "Embeddings-only enrollment. Descriptors are generated with OpenCV SFace ONNX in the browser using ONNX Runtime Web. Do not compare these with face-api faceRecognitionNet descriptors.",
      },
      captures: embeddings,
    };
  }

  private assertValidPayload(payload: SFaceEmbeddingPayload): void {
    if (!payload.captures.length) throw new Error("No embeddings were captured.");
    for (const capture of payload.captures) {
      if (!Array.isArray(capture.descriptor) || capture.descriptor.length !== 128) {
        throw new Error(`Invalid SFace descriptor for '${capture.label}': expected 128 numbers.`);
      }
      if (!capture.descriptor.every((value) => Number.isFinite(value))) {
        throw new Error(`Invalid SFace descriptor for '${capture.label}': descriptor contains non-finite values.`);
      }
      const norm = descriptorNorm(capture.descriptor);
      if (Math.abs(norm - 1.0) > 0.05) {
        throw new Error(`Invalid SFace descriptor for '${capture.label}': expected L2 norm close to 1.0, got ${norm.toFixed(4)}.`);
      }
    }
  }
}
