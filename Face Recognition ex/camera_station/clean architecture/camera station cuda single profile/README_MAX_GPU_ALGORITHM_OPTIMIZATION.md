# Max GPU / algorithm optimization patch

This patch keeps the UI and manual zoom behavior, but optimizes the recognition loop itself.

## Main fix

Previously, when a profile skipped detection/SFace on cached frames, the desktop worker still reprocessed the old cached candidates on CPU every frame. That made profiles look almost identical in Task Manager because the CPU kept doing tracking, unknown handling, known counters, and report work even when detection was skipped.

Now cached frames are display-only by default:

- fresh detection frames run YuNet + SFace + tracking/counting;
- cached frames reuse the last candidates for display only;
- known/unknown counters are not incremented from stale candidates;
- unknown JSON/image registry is not touched on cached frames;
- UI controls are synced at UI cadence, not every camera frame.

## New profile

`gpu_batch_max`

This profile is designed for your confirmed ONNX Runtime CUDA setup:

- camera: 960x540
- YuNet input: 512 px
- detection every 2 frames
- SFace max recognitions/frame: 0 = all unresolved faces in one CUDA batch
- unknown drawing OFF
- unknown registry updates OFF
- stable frames: 1 for faster walk-through labelling
- cached-frame processing OFF

Use it when CUDA benchmark works and you want to keep GPU fed with larger SFace batches.

## Recommended settings

Profile: `gpu_batch_max`

- Try GPU acceleration: ON
- Periodic grid search: OFF
- Draw unknown faces: OFF
- Skip recognition for resolved faces: ON
- Max SFace recognitions/frame: 0 for maximum batching, or 24/32 if the CPU alignment gets too high
- Stop repeated unknown faces: ON

## How to verify the fix

In the live metrics line you should see:

`det every 2 fresh 1` on detection frames

and:

`det every 2 fresh 0` on cached display frames.

On fresh=0 frames, `track` and `loop` should drop strongly compared with previous builds.

If CPU is still high, check the timing fields:

- `det` high = YuNet CPU detector is the bottleneck;
- `align` high = SFace alignment/crop is CPU bottleneck;
- `infer` high = GPU SFace inference bottleneck;
- `draw` high = UI/drawing bottleneck;
- `read` high = camera/driver/stream bottleneck.

The next major speed step after this patch is moving YuNet detection itself to ONNX Runtime CUDA or replacing YuNet with a GPU-first detector.
