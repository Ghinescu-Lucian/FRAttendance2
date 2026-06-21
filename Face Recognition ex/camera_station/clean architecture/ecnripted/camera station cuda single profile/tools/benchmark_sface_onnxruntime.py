"""Benchmark the SFace ONNX model with ONNX Runtime CPU vs CUDA.

Run from the opencv_station folder:
    py tools\\benchmark_sface_onnxruntime.py --model models\\face_recognition_sface_2021dec_ort_dynamic.onnx

This tests only neural-network inference. It does not include camera capture,
YuNet detection, Tkinter UI, drawing, disk writes, or OpenCV alignCrop.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def _shape_to_nchw(shape, batch: int) -> tuple[int, int, int, int]:
    shape = list(shape or [])
    if len(shape) != 4:
        return (batch, 3, 112, 112)
    c = shape[1] if isinstance(shape[1], int) and shape[1] > 0 else 3
    h = shape[2] if isinstance(shape[2], int) and shape[2] > 0 else 112
    w = shape[3] if isinstance(shape[3], int) and shape[3] > 0 else 112
    return (batch, int(c), int(h), int(w))


def _fixed_batch(shape) -> int | None:
    shape = list(shape or [])
    if len(shape) >= 1 and isinstance(shape[0], int) and shape[0] > 0:
        return int(shape[0])
    return None


def bench(model_path: Path, provider: str, batch_sizes: list[int], warmup: int, repeat: int) -> None:
    if provider not in ort.get_available_providers():
        print(f"\n[{provider}] NOT AVAILABLE. Available providers: {ort.get_available_providers()}")
        return

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_mem_pattern = True
    options.enable_cpu_mem_arena = True

    providers = [(provider, {"device_id": 0})] if provider == "CUDAExecutionProvider" else [provider]
    if provider != "CPUExecutionProvider":
        providers.append("CPUExecutionProvider")

    session = ort.InferenceSession(str(model_path), sess_options=options, providers=providers)
    actual = session.get_providers()
    input_meta = session.get_inputs()[0]
    output_name = session.get_outputs()[0].name
    fixed = _fixed_batch(input_meta.shape)

    print(f"\n[{provider}] actual providers: {actual}")
    print(f"Input: name={input_meta.name}, shape={input_meta.shape}")
    if fixed is not None:
        print(f"WARNING: model has fixed batch={fixed}. Batches different from {fixed} will be skipped.")
        print("         Create the dynamic model first: py tools\\make_sface_dynamic_batch.py --input models\\face_recognition_sface_2021dec.onnx")

    for batch in batch_sizes:
        if fixed is not None and batch != fixed:
            print(f"batch={batch:>2}  skipped: fixed model expects batch={fixed}")
            continue

        nchw = _shape_to_nchw(input_meta.shape, batch)
        x = np.random.rand(*nchw).astype(np.float32)
        feed = {input_meta.name: x}

        for _ in range(warmup):
            session.run([output_name], feed)

        times_ms = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            session.run([output_name], feed)
            times_ms.append((time.perf_counter() - t0) * 1000.0)

        avg = statistics.mean(times_ms)
        p50 = statistics.median(times_ms)
        fps_faces = batch * 1000.0 / avg if avg > 0 else 0.0
        print(f"batch={batch:>2}  avg={avg:>8.3f} ms  p50={p50:>8.3f} ms  throughput={fps_faces:>8.1f} faces/s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/face_recognition_sface_2021dec_ort_dynamic.onnx")
    parser.add_argument("--batches", default="1,4,8,16,32")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        fallback = Path("models/face_recognition_sface_2021dec.onnx")
        print(f"Model not found: {model_path.resolve()}")
        print(f"Falling back to: {fallback.resolve()}")
        model_path = fallback
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path.resolve()}")

    batch_sizes = [int(x.strip()) for x in args.batches.split(",") if x.strip()]
    print("ONNX Runtime:", ort.__version__)
    print("Available providers:", ort.get_available_providers())
    print("Model:", model_path.resolve())

    bench(model_path, "CPUExecutionProvider", batch_sizes, args.warmup, args.repeat)
    bench(model_path, "CUDAExecutionProvider", batch_sizes, args.warmup, args.repeat)


if __name__ == "__main__":
    main()
