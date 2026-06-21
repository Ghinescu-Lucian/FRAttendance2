import { ActiveSession, KnownFaceEmbedding, RecognitionCandidate, RecognitionResult } from "../../domain/types";
import { RECOGNITION_RULES } from "../../shared/appConfig";
import { cosineSimilarity, euclideanDistance } from "../../shared/math";

export class RecognitionApi {
  private knownFaces: KnownFaceEmbedding[] = [];
  private activeSession: ActiveSession | null = null;

  async refreshActiveSession(): Promise<ActiveSession | null> {
    const context = this.getMoodleContextOrNull();
    if (!context?.activeSessionUrl) return null;

    const response = await fetch(context.activeSessionUrl, { cache: "no-store", credentials: "same-origin" });
    const data = await response.json().catch(() => null);
    if (!response.ok || !data?.ok) {
      throw new Error(data?.error || `Could not load active session. Status: ${response.status}`);
    }

    this.activeSession = data.active ? data.session : null;
    return this.activeSession;
  }

  getActiveSession(): ActiveSession | null {
    return this.activeSession;
  }

  async refreshKnownFaces(): Promise<KnownFaceEmbedding[]> {
    const context = this.getMoodleContextOrNull();
    const url = context?.knownFacesUrl || "/api/known-faces";

    const response = await fetch(url, { cache: "no-store", credentials: "same-origin" });
    const data = await response.json().catch(() => null);

    if (!response.ok || !data?.ok) {
      throw new Error(data?.error || `Could not load known faces. Status: ${response.status}`);
    }

    this.knownFaces = Array.isArray(data.faces) ? data.faces : [];
    return this.knownFaces;
  }

  getKnownFaceCount(): number {
    return this.knownFaces.length;
  }

  recognize(descriptor: number[]): RecognitionResult {
    let best: RecognitionCandidate | null = null;

    for (const face of this.knownFaces) {
      if (!Array.isArray(face.descriptor) || face.descriptor.length !== descriptor.length) continue;

      const distance = euclideanDistance(descriptor, face.descriptor);
      const similarity = cosineSimilarity(descriptor, face.descriptor);

      if (!best || similarity > best.similarity) {
        best = {
          userid: face.userid,
          studentId: face.studentId,
          name: face.name,
          distance,
          similarity,
          sourceFile: face.sourceFile,
          pose: face.pose,
        };
      }
    }

    const matched = Boolean(
      best &&
      best.distance <= RECOGNITION_RULES.matchDistanceThreshold &&
      best.similarity >= RECOGNITION_RULES.matchSimilarityThreshold,
    );

    return {
      matched,
      best,
      candidatesChecked: this.knownFaces.length,
    };
  }

  async markKnownFace(result: RecognitionResult, source = "browser-camera-station"): Promise<void> {
    const context = this.getMoodleContextOrNull();
    if (!context?.markDetectionUrl || !this.activeSession || !result.best) return;

    const userid = result.best.userid ?? Number(result.best.studentId);
    if (!Number.isFinite(userid) || userid <= 0) return;

    const response = await fetch(context.markDetectionUrl, {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        cmid: context.cmid,
        sesskey: context.sesskey,
        sessionid: this.activeSession.id,
        userid,
        confidence: result.best.similarity,
        distance: result.best.distance,
        source,
      }),
    });

    const data = await response.json().catch(() => null);
    if (!response.ok || !data?.ok) {
      throw new Error(data?.error || `Could not mark attendance. Status: ${response.status}`);
    }
  }

  async saveUnknownFace(descriptor: number[], result: RecognitionResult, source = "browser-camera-station", thumbnail?: string | null): Promise<void> {
    const context = this.getMoodleContextOrNull();
    if (!context?.saveUnknownUrl || !this.activeSession) return;

    const response = await fetch(context.saveUnknownUrl, {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        cmid: context.cmid,
        sesskey: context.sesskey,
        sessionid: this.activeSession.id,
        descriptor,
        candidate: result.best,
        source,
        thumbnail,
      }),
    });

    const data = await response.json().catch(() => null);
    if (!response.ok || !data?.ok) {
      throw new Error(data?.error || `Could not save unknown face. Status: ${response.status}`);
    }
  }

  async saveCaptureFace(descriptor: number[], result: RecognitionResult, source = "browser-capture-intake", thumbnail?: string | null): Promise<any> {
    const context = this.getMoodleContextOrNull();
    if (!context?.captureFaceUrl || !context?.captureSessionId) return null;

    const response = await fetch(context.captureFaceUrl, {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        cmid: context.cmid,
        sesskey: context.sesskey,
        capturesessionid: context.captureSessionId,
        descriptor,
        candidate: result.best,
        source,
        thumbnail,
      }),
    });

    const data = await response.json().catch(() => null);
    if (!response.ok || !data?.ok) {
      throw new Error(data?.error || `Could not capture entering face. Status: ${response.status}`);
    }

    return data;
  }

  private getMoodleContextOrNull(): FaceAttendanceMoodleContext | null {
    return window.FACEATTENDANCE_CONTEXT || null;
  }
}
