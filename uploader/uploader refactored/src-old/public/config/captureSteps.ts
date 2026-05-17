import { CaptureStep } from "../types.js";

export const CAPTURE_STEPS: CaptureStep[] = [
  {
    pose: "front",
    label: "Front",
    instruction: "Look straight at the camera.",
    speech: "Look straight at the camera.",
    isPoseOk: (q) => Math.abs(q.yaw) < 0.12 && Math.abs(q.pitch) < 0.12,
  },
  {
    pose: "right",
    label: "Head right",
    instruction: "Turn your head slightly to YOUR right.",
    speech: "Turn your head slightly to your right.",
    isPoseOk: (q) => q.yaw > 0.10,
  },
  {
    pose: "left",
    label: "Head left",
    instruction: "Turn your head slightly to YOUR left.",
    speech: "Turn your head slightly to your left.",
    isPoseOk: (q) => q.yaw < -0.10,
  },
  {
    pose: "down",
    label: "Head down",
    instruction: "Tilt your head slightly down.",
    speech: "Tilt your head slightly down.",
    isPoseOk: (q) => q.pitch > 0.06,
  },
  {
    pose: "front_2",
    label: "Front again",
    instruction: "Look straight again.",
    speech: "Look straight again.",
    isPoseOk: (q) => Math.abs(q.yaw) < 0.12 && Math.abs(q.pitch) < 0.12,
  },
];
