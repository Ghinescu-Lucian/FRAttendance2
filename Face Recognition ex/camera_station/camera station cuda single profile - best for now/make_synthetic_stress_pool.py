#!/usr/bin/env python3
"""
Create random normalized 128D vectors in your station's JSON format.

IMPORTANT:
  These are NOT real face embeddings. Use them only to stress-test loader speed,
  memory usage, and recognition-loop scaling. Do not use this for accuracy tests.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Create synthetic random SFace-like vectors for performance stress tests.")
    parser.add_argument("--people", type=int, default=1000)
    parser.add_argument("--embeddings-per-person", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("synthetic_stress_pool"))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--single-file", action="store_true", help="Write one combined JSON instead of one file per person.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    def random_embeddings(count: int):
        mat = rng.normal(size=(count, 128)).astype(np.float32)
        mat /= np.linalg.norm(mat, axis=1, keepdims=True)
        return mat.astype(float).tolist()

    model = {
        "family": "opencv",
        "detector": "synthetic-none",
        "recognizer": "sface-compatible-shape-only",
        "descriptorLength": 128,
        "descriptorNormalized": True,
        "warning": "Random vectors for performance tests only; not real face embeddings.",
    }

    if args.single_file:
        combined = {
            "version": 1,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": model,
            "source": "synthetic_random_vectors_for_stress_testing_only",
            "people": [],
        }
        for i in range(args.people):
            combined["people"].append({
                "name": f"Synthetic_{i + 1:05d}",
                "embeddings": random_embeddings(args.embeddings_per_person),
            })
        out = args.output / "synthetic_stress_pool.json"
        out.write_text(json.dumps(combined), encoding="utf-8")
        print(f"[DONE] Wrote {args.people * args.embeddings_per_person} synthetic vectors -> {out}")
        return 0

    total = 0
    for i in range(args.people):
        payload = {
            "version": 1,
            "name": f"Synthetic_{i + 1:05d}",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": model,
            "source": "synthetic_random_vectors_for_stress_testing_only",
            "embeddings": random_embeddings(args.embeddings_per_person),
        }
        out = args.output / f"Synthetic_{i + 1:05d}_sface_embeddings.json"
        out.write_text(json.dumps(payload), encoding="utf-8")
        total += args.embeddings_per_person
    print(f"[DONE] Wrote {total} synthetic vectors in {args.people} files -> {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
