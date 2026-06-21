# CPU bottleneck / real profile difference patch

This patch keeps SFace on ONNX Runtime CUDA, but reduces CPU work in the desktop pipeline.

## Main fix

Before this patch, when a profile skipped detection on intermediate frames, the desktop worker still reused the previous candidates and repeated CPU work every frame:

- unknown tracking;
- unknown registry checks;
- known report counters;
- stability counters;
- cleanup logic.

That made `crowd_fast`, `crowd_turbo`, and `crowd_extreme` look almost the same in Task Manager, because only YuNet/SFace were skipped, not the rest of the candidate processing.

Now stale/reused candidates are normally used only for display. They are not recounted or reprocessed unless `FACEATTENDANCE_DESKTOP_PROCESS_STALE_CANDIDATES=true` is set.

## Additional changes

- Tk/control refresh is throttled instead of being read every camera frame.
- The live metrics now show `det every N` and `fresh 0/1`.
- `walkthrough_realtime` is made more CPU-balanced:
  - YuNet every 2 frames instead of every frame;
  - YuNet input width 512 instead of 640;
  - top_k reduced from 500 to 300;
  - stability remains 2 detection hits for moving people.

## Recommended tests

For people walking normally:

```text
Profile: walkthrough_realtime
Try GPU acceleration: ON
Draw unknown faces: OFF
Periodic grid search: OFF
Max SFace recognitions/frame: 16-24
```

For many people in a class/crowd:

```text
Profile: crowd_extreme
Try GPU acceleration: ON
Draw unknown faces: OFF
Periodic grid search: OFF
```

Stop and start the station after changing profile, because camera resolution and detector cadence are set when the worker starts.
