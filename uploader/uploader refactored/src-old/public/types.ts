export interface QualityInfo {
  score: number;
  blur: number;
  brightness: number;
  faceHeightRatio: number;
  centered: boolean;
  yaw: number;
  pitch: number;
}

export interface CapturedEmbedding {
  studentId: string;
  name: string;
  pose: string;
  label: string;
  descriptor: number[];
  quality: QualityInfo;
  capturedAt: string;
}

export interface CaptureStep {
  pose: string;
  label: string;
  instruction: string;
  speech: string;
  isPoseOk: (quality: QualityInfo) => boolean;
}

export interface Point2D {
  x: number;
  y: number;
}

export interface StudentIdentity {
  studentId: string;
  personName: string;
}

export interface SFaceEmbeddingPayload {
  version: number;
  studentId: string;
  name: string;
  createdAt: string;
  model: {
    family: string;
    detectorForEnrollment: string;
    recognizer: string;
    recognizerModel: string;
    descriptorLength: number;
    descriptorNormalized: boolean;
    alignedSize: number[];
    clientBuild: string;
    note: string;
  };
  captures: CapturedEmbedding[];
}
