import { useEffect, useRef } from "react";
import { RecognitionController } from "../application/recognition/RecognitionController";
import { CameraService } from "../infrastructure/browser/CameraService";
import { FaceDetectionService } from "../infrastructure/ml/FaceDetectionService";
import { SFaceEmbeddingService } from "../infrastructure/ml/SFaceEmbeddingService";
import { RecognitionApi } from "../infrastructure/api/RecognitionApi";
import { RecognitionView } from "./view/RecognitionView";

function useRecognitionController(): void {
  const controllerRef = useRef<RecognitionController | null>(null);

  useEffect(() => {
    const view = new RecognitionView();
    const controller = new RecognitionController(
      view,
      new CameraService(view.video),
      new FaceDetectionService(),
      new SFaceEmbeddingService(),
      new RecognitionApi(),
    );

    controller.bind();
    controllerRef.current = controller;

    return () => {
      controller.stop();
      controllerRef.current = null;
    };
  }, []);
}

export function FaceRecognitionPage() {
  useRecognitionController();
  const isCaptureMode = window.FACEATTENDANCE_CONTEXT?.mode === "capture";

  return (
    <main className="enrollment-page">
      <section className="hero-card">
        <div>
          <p className="eyebrow">{isCaptureMode ? "Capture-first intake" : "Local SFace recognition"}</p>
          <h1>{isCaptureMode ? "Capture entering faces" : "Facial recognition"}</h1>
          <p className="description">
            {isCaptureMode
              ? "Detect every entering face, group repeated observations, and store at most two temporary images for teacher labeling later."
              : "Load enrolled SFace embeddings from the server, detect a live face, generate an embedding locally, and compare it with the enrolled database."}
          </p>
        </div>
      </section>

      <section className="panel controls-panel recognition-controls" aria-label="Recognition controls">
        <label className="field camera-select-field recognition-camera-select">
          <span>Camera</span>
          <select id="recognitionCameraSelect" defaultValue="">
            <option value="">Default camera</option>
          </select>
        </label>

        <div className="button-row">
          <button id="refreshRecognitionCamerasBtn" type="button" className="secondary-btn">Refresh cameras</button>
          <button id="startRecognitionCameraBtn" type="button" className="primary-btn">{isCaptureMode ? "Start capture" : "Start recognition"}</button>
          <button id="refreshKnownFacesBtn" type="button" className="secondary-btn">Refresh enrolled faces</button>
          <button id="stopRecognitionCameraBtn" type="button" className="danger-btn" disabled>{isCaptureMode ? "Stop capture" : "Stop recognition"}</button>
        </div>
      </section>

      <section className="camera-grid">
        <div className="camera-card">
          <div className="camera-frame">
            <video id="recognitionVideo" autoPlay muted playsInline />
            <canvas id="recognitionOverlayCanvas" />
            <canvas id="recognitionCanvas" />
          </div>
        </div>

        <aside className="panel status-panel recognition-panel">
          <h2>{isCaptureMode ? "Capture status" : "Recognition status"}</h2>
          <p id="knownFacesCount" className="status-text">Known embeddings loaded: 0</p>
          <p id="recognitionStatus" className="status-text">Waiting.</p>
          <p id="recognitionResult" className="recognition-result neutral">{isCaptureMode ? "No captured group yet." : "No recognition result yet."}</p>
        </aside>
      </section>
    </main>
  );
}
