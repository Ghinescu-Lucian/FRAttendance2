"""Create a dynamic-batch ONNX Runtime copy of the OpenCV Zoo SFace model.

Run from the opencv_station folder:
    py tools\\make_sface_dynamic_batch.py --input models\\face_recognition_sface_2021dec.onnx

Output:
    models\\face_recognition_sface_2021dec_ort_dynamic.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path


def set_symbolic_batch(value_info, batch_name: str = "N") -> None:
    tensor_type = value_info.type.tensor_type
    shape = tensor_type.shape
    if len(shape.dim) >= 1:
        shape.dim[0].ClearField("dim_value")
        shape.dim[0].dim_param = batch_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="models/face_recognition_sface_2021dec.onnx")
    parser.add_argument("--output", default=None)
    parser.add_argument("--batch-name", default="N")
    args = parser.parse_args()

    try:
        import onnx
    except Exception as exc:
        raise SystemExit("Missing dependency. Install it with: py -m pip install onnx") from exc

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input model not found: {input_path.resolve()}")

    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_ort_dynamic{input_path.suffix}")

    model = onnx.load(str(input_path))

    initializer_names = {init.name for init in model.graph.initializer}
    old_inputs = list(model.graph.input)
    new_inputs = [value_info for value_info in old_inputs if value_info.name not in initializer_names]
    removed = len(old_inputs) - len(new_inputs)
    if removed > 0:
        del model.graph.input[:]
        model.graph.input.extend(new_inputs)

    for value_info in list(model.graph.input) + list(model.graph.output):
        set_symbolic_batch(value_info, args.batch_name)

    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))

    print(f"Created: {output_path.resolve()}")
    print(f"Removed initializer graph inputs: {removed}")
    print(f"Enabled symbolic batch dimension: {args.batch_name}")


if __name__ == "__main__":
    main()
