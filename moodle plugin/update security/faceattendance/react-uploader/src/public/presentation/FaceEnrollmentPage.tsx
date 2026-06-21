import { useEffect, useMemo, useRef } from "react";
import { EnrollmentController } from "../application/enrollment/EnrollmentController";
import { InputService } from "../application/enrollment/InputService";
import { QualityService } from "../application/enrollment/QualityService";
import { CameraService } from "../infrastructure/browser/CameraService";
import { SpeechService } from "../infrastructure/browser/SpeechService";
import { EnrollmentApi } from "../infrastructure/api/EnrollmentApi";
import { FaceDetectionService } from "../infrastructure/ml/FaceDetectionService";
import { SFaceEmbeddingService } from "../infrastructure/ml/SFaceEmbeddingService";
import { EnrollmentView } from "./view/EnrollmentView";
import { installPoseOverlay } from "./view/poseOverlay";

function useEnrollmentController(): void {
  const controllerRef = useRef<EnrollmentController | null>(null);

  useEffect(() => {
    const view = new EnrollmentView();
    installPoseOverlay(view);

    const controller = new EnrollmentController(
      view,
      new InputService(view),
      new CameraService(view.video),
      new FaceDetectionService(),
      new QualityService(),
      new SFaceEmbeddingService(),
      new EnrollmentApi(),
      new SpeechService(),
    );

    controller.bind();
    controllerRef.current = controller;

    return () => {
      controller.stopCamera();
      controllerRef.current = null;
    };
  }, []);
}

export function FaceEnrollmentPage() {
  useEnrollmentController();

  const context = window.FACEATTENDANCE_CONTEXT;
  const buildLabel = useMemo(() => context ? "Moodle Face Attendance enrollment" : "Local SFace enrollment", [context]);
  const description = context
    ? "Capture guided poses, generate SFace embeddings locally in the browser, then save the embedding JSON directly inside Moodle. No raw face images are uploaded."
    : "Capture guided poses, generate SFace embeddings locally in the browser, then upload one JSON file to the server.";

  return (
    <main className="enrollment-page">
      <section className="hero-card">
        <div>
          <p className="eyebrow">{buildLabel}</p>
          <h1>Student face enrollment</h1>
          <p className="description">{description}</p>
        </div>
      </section>

      <section className="panel controls-panel" aria-label="Enrollment controls">
        <label className="field">
          <span>Student ID</span>
          <input
            id="studentId"
            type="text"
            placeholder="e.g. 1001"
            autoComplete="off"
            defaultValue={context?.studentId || ""}
            readOnly={Boolean(context)}
          />
        </label>

        <label className="field">
          <span>Student name</span>
          <input
            id="personName"
            type="text"
            placeholder="e.g. Ghinescu Lucian"
            autoComplete="name"
            defaultValue={context?.studentName || ""}
            readOnly={Boolean(context)}
          />
        </label>


        <label className="field camera-select-field">
          <span>Camera</span>
          <select id="cameraSelect" defaultValue="">
            <option value="">Default camera</option>
          </select>
        </label>

        <div className="button-row">
          <button id="refreshCamerasBtn" type="button" className="secondary-btn">Refresh cameras</button>
          <button id="startCameraBtn" type="button" className="primary-btn">Start camera</button>
          <button id="captureBtn" type="button" className="secondary-btn" disabled>Capture embeddings</button>
          <button id="stopCameraBtn" type="button" className="danger-btn" disabled>Stop camera</button>
          {context?.returnUrl ? <a className="return-link" href={context.returnUrl}>Back to Moodle registration page</a> : null}
        </div>
      </section>

      <section className="camera-grid">
        <div className="camera-card">
          <div className="camera-frame">
            <video id="video" autoPlay muted playsInline />
            <canvas id="overlayCanvas" />
            <canvas id="canvas" />
          </div>
        </div>

        <aside className="panel status-panel">
          <h2>Capture status</h2>
          <p id="step" className="step-text">Ready.</p>
          <p id="status" className="status-text">Waiting.</p>
        </aside>
      </section>

      <section className="panel previews-panel">
        <h2>Accepted captures</h2>
        <div id="previews" className="previews" />
      </section>
    </main>
  );
}
