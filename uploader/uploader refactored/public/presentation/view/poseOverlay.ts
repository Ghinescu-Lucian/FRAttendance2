import { EnrollmentView } from "./EnrollmentView";

export function installPoseOverlay(view: EnrollmentView): void {
  const host = view.video.parentElement;
  if (!host || document.getElementById("poseOverlayStyles")) return;

  const style = document.createElement("style");
  style.id = "poseOverlayStyles";
  style.textContent = `
    .pose-overlay-host {
      position: relative !important;
      overflow: hidden;
    }

    .pose-overlay-box {
      position: absolute;
      left: 12px;
      right: 12px;
      top: 10px;
      z-index: 40;
      pointer-events: none;
      padding: 8px 10px;
      border-radius: 12px;
      background: rgba(2, 6, 23, 0.72);
      border: 1px solid rgba(148, 163, 184, 0.35);
      backdrop-filter: blur(6px);
      -webkit-backdrop-filter: blur(6px);
      box-sizing: border-box;
    }

    .pose-overlay-box #step {
      margin: 0 !important;
      color: #fff !important;
      font-size: 15px !important;
      font-weight: 800 !important;
      line-height: 1.25 !important;
      text-align: center;
      text-shadow: 0 1px 3px rgba(0,0,0,.85);
    }
  `;

  document.head.appendChild(style);
  host.classList.add("pose-overlay-host");

  const overlay = document.createElement("div");
  overlay.id = "poseOverlayText";
  overlay.className = "pose-overlay-box";
  host.appendChild(overlay);
  overlay.appendChild(view.stepEl);
}
