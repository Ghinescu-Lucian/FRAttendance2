import { ENROLLMENT_APP_BUILD, SFACE_SIZE } from "../../shared/appConfig";
import { CapturedEmbedding, SFaceEmbeddingPayload, StudentIdentity } from "../../domain/types";
import { descriptorNorm } from "../../shared/math";

export class EnrollmentApi {
  private lastUploadSummary: any | null = null;

  async uploadImage(_identity: StudentIdentity, _blob: Blob, _frameIndex: number, _pose: string): Promise<void> {
    throw new Error("Image uploads are disabled. This Moodle integration stores only embedding JSON data.");
  }

  async uploadEmbeddingFile(identity: StudentIdentity, embeddings: CapturedEmbedding[]): Promise<void> {
    const payload = this.buildPayload(identity, embeddings);
    this.assertValidPayload(payload);

    const context = this.getMoodleContext();
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const formData = new FormData();

    formData.append("cmid", String(context.cmid));
    formData.append("userid", String(context.userid));
    formData.append("sesskey", context.sesskey);
    formData.append("studentId", identity.studentId);
    formData.append("name", identity.personName);
    formData.append("model", "opencv-sface-2021dec");
    formData.append("clientBuild", ENROLLMENT_APP_BUILD);
    formData.append("embeddingFile", blob, `${identity.personName}_${identity.studentId}_sface_embeddings.json`);

    const response = await fetch(context.saveEmbeddingUrl, {
      method: "POST",
      body: formData,
      credentials: "same-origin",
    });

    const data = await response.json().catch(() => null);
    if (!response.ok || !data?.ok) {
      const details = Array.isArray(data?.details) ? ` ${data.details.join(" ")}` : "";
      throw new Error((data?.error || `Embedding save failed with status ${response.status}`) + details);
    }

    this.lastUploadSummary = data;
  }

  async completeEnrollment(_identity: StudentIdentity): Promise<any> {
    return this.lastUploadSummary || {
      ok: true,
      savedImages: 0,
      savedEmbeddingFiles: 1,
      uploadDir: "Moodle database",
    };
  }

  private getMoodleContext(): FaceAttendanceMoodleContext {
    const context = window.FACEATTENDANCE_CONTEXT;
    if (!context) {
      throw new Error("Missing Moodle context. Open this page from the Face Attendance plugin recorder page.");
    }
    return context;
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
