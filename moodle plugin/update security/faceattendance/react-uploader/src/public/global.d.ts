declare const faceapi: any;
declare const ort: any;

interface FaceAttendanceMoodleContext {
  mode?: "enrollment" | "station" | "capture";
  cmid: number;
  userid: number;
  sesskey: string;
  studentId: string;
  studentName: string;
  saveEmbeddingUrl?: string;
  sfaceModelUrl: string;
  knownFacesUrl?: string;
  activeSessionUrl?: string;
  markDetectionUrl?: string;
  saveUnknownUrl?: string;
  captureFaceUrl?: string;
  captureSessionId?: number;
  returnUrl?: string;
  stationProfile?: string;
  stationProfileLabel?: string;
}

interface Window {
  FACEATTENDANCE_CONTEXT?: FaceAttendanceMoodleContext;
}
