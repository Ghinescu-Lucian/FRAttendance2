import { KnownFaceEmbedding, RecognitionCandidate, RecognitionResult } from "../../domain/types";
import { RECOGNITION_RULES } from "../../shared/appConfig";
import { cosineSimilarity, euclideanDistance } from "../../shared/math";

export class RecognitionApi {
  private knownFaces: KnownFaceEmbedding[] = [];

  async refreshKnownFaces(): Promise<KnownFaceEmbedding[]> {
    const response = await fetch("/api/known-faces", { cache: "no-store" });
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

      if (!best || distance < best.distance) {
        best = {
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
}
