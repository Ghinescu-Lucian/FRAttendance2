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

  return (
    <main className="enrollment-page">
      <section className="hero-card">
        <div>
          <p className="eyebrow">Local SFace recognition</p>
          <h1>Facial recognition</h1>
          <p className="description">
            Load enrolled SFace embeddings from the server, detect a live face, generate an embedding locally, and compare it with the enrolled database.
          </p>
        </div>
      </section>

      <section className="panel controls-panel recognition-controls" aria-label="Recognition controls">
        <div className="button-row">
          <button id="startRecognitionCameraBtn" type="button" className="primary-btn">Start recognition</button>
          <button id="refreshKnownFacesBtn" type="button" className="secondary-btn">Refresh enrolled faces</button>
          <button id="stopRecognitionCameraBtn" type="button" className="danger-btn" disabled>Stop recognition</button>
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
          <h2>Recognition status</h2>
          <p id="knownFacesCount" className="status-text">Known embeddings loaded: 0</p>
          <p id="recognitionStatus" className="status-text">Waiting.</p>
          <p id="recognitionResult" className="recognition-result neutral">No recognition result yet.</p>
        </aside>
      </section>
    </main>
  );
}
