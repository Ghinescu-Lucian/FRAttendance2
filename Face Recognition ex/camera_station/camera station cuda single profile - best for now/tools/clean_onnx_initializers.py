"""Create an ONNX Runtime-friendly copy by removing initializer tensors from graph inputs.

Run from opencv_station:
    py -m pip install onnx
    py tools\\clean_onnx_initializers.py --input models\\face_recognition_sface_2021dec.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx


def clean(input_path: Path, output_path: Path) -> int:
    model = onnx.load(str(input_path))
    initializer_names = {init.name for init in model.graph.initializer}
    old_inputs = list(model.graph.input)
    new_inputs = [value_info for value_info in old_inputs if value_info.name not in initializer_names]
    removed = len(old_inputs) - len(new_inputs)
    if removed:
        del model.graph.input[:]
        model.graph.input.extend(new_inputs)
    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))
    return removed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="models/face_recognition_sface_2021dec.onnx")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(f"{input_path.stem}_ort_clean{input_path.suffix}")

    if not input_path.exists():
        raise SystemExit(f"Input model not found: {input_path.resolve()}")

    removed = clean(input_path, output_path)
    print(f"Saved: {output_path.resolve()}")
    print(f"Removed initializer graph inputs: {removed}")


if __name__ == "__main__":
    main()
