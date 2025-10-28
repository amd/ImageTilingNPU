#!/usr/bin/env python3
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Inject input/output stride shapes into a VOE vaip_config.json target.

Reads ONNX model I/O shapes, computes SR stride from image size, patches the
named target in voe_package/bin/vaip_config.json to match SuperResolution
vaip_config_stride.json X1-Benchmark block, writes vaip_config/vaip_config_stride.json.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT_DIR / "voe_package" / "bin" / "vaip_config.json"
DEFAULT_OUTPUT = ROOT_DIR / "vaip_config" / "vaip_config_stride.json"
DEFAULT_TARGET = "X1-Benchmark"


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_hwc_shape(shape: list[Any], tensor_name: str) -> tuple[int, int, int]:
    """Return (H, W, C) from [N,H,W,C] or [N,C,H,W]."""
    if len(shape) != 4:
        raise ValueError(f"{tensor_name}: expected 4D shape, got {shape}")

    dims: list[int] = []
    for dim in shape:
        if isinstance(dim, str):
            raise ValueError(f"{tensor_name}: dynamic dimension '{dim}' is not supported")
        dims.append(int(dim))

    n, d1, d2, d3 = dims
    if n != 1:
        raise ValueError(f"{tensor_name}: batch size must be 1, got {n}")

    if d3 in (3, 4):
        return d1, d2, d3
    if d1 in (3, 4):
        return d2, d3, d1
    raise ValueError(f"{tensor_name}: cannot infer HWC layout from shape {shape}")


def read_onnx_hwc_shapes(model_path: Path) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Return ((input_h, input_w, input_c), (output_h, output_w, output_c))."""
    try:
        import onnx

        model = onnx.load(str(model_path))
        if not model.graph.input or not model.graph.output:
            raise ValueError("ONNX model missing input or output")

        in_tensor = model.graph.input[0]
        out_tensor = model.graph.output[0]
        in_shape = [d.dim_value or d.dim_param for d in in_tensor.type.tensor_type.shape.dim]
        out_shape = [d.dim_value or d.dim_param for d in out_tensor.type.tensor_type.shape.dim]
        input_hwc = parse_hwc_shape(in_shape, in_tensor.name)
        output_hwc = parse_hwc_shape(out_shape, out_tensor.name)
        return input_hwc, output_hwc
    except ImportError:
        pass

    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        in_shape = session.get_inputs()[0].shape
        out_shape = session.get_outputs()[0].shape
        input_hwc = parse_hwc_shape(in_shape, session.get_inputs()[0].name)
        output_hwc = parse_hwc_shape(out_shape, session.get_outputs()[0].name)
        return input_hwc, output_hwc
    except ImportError as exc:
        raise ImportError("install onnx or onnxruntime to read ONNX shapes") from exc


def compute_stride_shapes(
    image_width: int,
    image_height: int,
    input_h: int,
    input_w: int,
    output_h: int,
    output_w: int,
    channels: int,
) -> tuple[list[int], list[int], dict[str, int]]:
    if input_h < output_h or input_w < output_w:
        raise ValueError(
            f"model input must be >= output: input {input_w}x{input_h}, output {output_w}x{output_h}"
        )

    pad_x = input_w - output_w
    pad_y = input_h - output_h
    output_stride_h = math.ceil(image_height / output_h) * output_h
    output_stride_w = math.ceil(image_width / output_w) * output_w
    input_stride_h = output_stride_h + pad_y
    input_stride_w = output_stride_w + pad_x

    meta = {
        "pad_x": pad_x,
        "pad_y": pad_y,
        "block_x": math.ceil(image_width / output_w),
        "block_y": math.ceil(image_height / output_h),
        "output_stride_h": output_stride_h,
        "output_stride_w": output_stride_w,
        "input_stride_h": input_stride_h,
        "input_stride_w": input_stride_w,
    }
    input_stride = [1, input_stride_h, input_stride_w, channels]
    output_stride = [1, output_stride_h, output_stride_w, channels]
    return input_stride, output_stride, meta


def list_target_names(config: dict[str, Any]) -> list[str]:
    targets = config.get("targets")
    if not isinstance(targets, list):
        return []
    return [str(t.get("name")) for t in targets if isinstance(t, dict) and t.get("name")]


def find_target(config: dict[str, Any], target_name: str) -> dict[str, Any]:
    targets = config.get("targets")
    if not isinstance(targets, list):
        raise ValueError("vaip_config missing 'targets' array")
    for target in targets:
        if isinstance(target, dict) and target.get("name") == target_name:
            return target
    raise ValueError(
        f"target '{target_name}' not found; available targets: {list_target_names(config)}"
    )


def inject_stride(
    config: dict[str, Any],
    target_name: str,
    input_stride: list[int],
    output_stride: list[int],
) -> dict[str, Any]:
    """Patch target block to match SuperResolution vaip_config_stride.json layout."""
    out = copy.deepcopy(config)
    target = find_target(out, target_name)
    target_opts = target.setdefault("target_opts", {})
    xcompiler = target_opts.setdefault("xcompilerAttrs", {})
    xcompiler["enable_qdq_to_qxint"] = {"boolValue": True}
    xcompiler["force_mode"] = {"uintValue": 7}
    xcompiler["input_stride_shape"] = {"intValues": input_stride}
    xcompiler["output_stride_shape"] = {"intValues": output_stride}
    target["old_qdq"] = False
    provider_options = target.setdefault("provider_options", {})
    provider_options["xlnx_enable_old_qdq"] = "0"
    return out


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT_DIR / path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inject SR stride shapes into voe_package vaip_config target block."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"source vaip_config.json (default: {DEFAULT_INPUT.relative_to(ROOT_DIR)})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output path (default: {DEFAULT_OUTPUT.relative_to(ROOT_DIR)})",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=DEFAULT_TARGET,
        help=f"target name in targets[] (default: {DEFAULT_TARGET})",
    )
    parser.add_argument("--model", type=Path, required=False, help="ONNX model path")
    parser.add_argument("--image-width", type=int, required=False, help="input image width")
    parser.add_argument("--image-height", type=int, required=False, help="input image height")
    parser.add_argument("--input-stride", type=str, default="", help="override: 1,H,W,C")
    parser.add_argument("--output-stride", type=str, default="", help="override: 1,H,W,C")
    parser.add_argument("--list-targets", action="store_true", help="print target names and exit")
    args = parser.parse_args()

    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)

    if not input_path.is_file():
        raise FileNotFoundError(f"input not found: {input_path}")

    config = json.loads(input_path.read_text(encoding="utf-8"))

    if args.list_targets:
        for name in list_target_names(config):
            print(name)
        return 0

    if args.model is None or args.image_width is None or args.image_height is None:
        if not (args.input_stride and args.output_stride):
            parser.error("--model, --image-width and --image-height are required "
                           "(unless --input-stride and --output-stride are both set)")

    if args.input_stride and args.output_stride:
        input_stride = parse_int_list(args.input_stride)
        output_stride = parse_int_list(args.output_stride)
        meta = None
        model_path = None
        input_hwc = output_hwc = None
    else:
        model_path = resolve_path(args.model)
        if not model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        (input_h, input_w, channels), (output_h, output_w, out_c) = read_onnx_hwc_shapes(model_path)
        if out_c != channels:
            raise ValueError(f"input/output channel mismatch: {channels} vs {out_c}")

        input_stride, output_stride, meta = compute_stride_shapes(
            args.image_width,
            args.image_height,
            input_h,
            input_w,
            output_h,
            output_w,
            channels,
        )
        input_hwc = (input_h, input_w, channels)
        output_hwc = (output_h, output_w, out_c)

    updated = inject_stride(config, args.target, input_stride, output_stride)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(updated, indent=4), encoding="utf-8")

    print(f"vaip input  : {input_path}")
    print(f"target      : {args.target}")
    if model_path is not None:
        print(f"onnx model  : {model_path}")
        print(f"image       : {args.image_width}x{args.image_height}")
        print(f"model input : {input_hwc[1]}x{input_hwc[0]} c={input_hwc[2]}")
        print(f"model output: {output_hwc[1]}x{output_hwc[0]} c={output_hwc[2]}")
    if meta is not None:
        print(f"pad_x/pad_y : {meta['pad_x']}/{meta['pad_y']}")
        print(f"blocks      : {meta['block_x']}x{meta['block_y']}")
    print(f"input_stride_shape  : {input_stride}")
    print(f"output_stride_shape : {output_stride}")
    print("enable_qdq_to_qxint : true")
    print("force_mode          : 7")
    print("old_qdq             : false")
    print("xlnx_enable_old_qdq : 0")
    print(f"written     : {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
