#!/usr/bin/env python3
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Generate a calibration grid: 1 mm fine lines, vivid line every 10 mm (H+V)."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def mm_to_px(mm: float, dpi: float) -> int:
    return max(1, int(round(mm * dpi / 25.4)))


def build_pattern(
    width: int,
    height: int,
    minor_px: int,
    major_px: int,
    minor_thin: int = 1,
    major_thick: int = 3,
) -> np.ndarray:
    img = np.full((height, width, 3), 255, dtype=np.uint8)

    # 1 mm fine grid (light gray)
    minor_color = (210, 210, 210)
    for x in range(0, width, minor_px):
        x1 = min(x + minor_thin, width)
        img[:, x:x1] = minor_color
    for y in range(0, height, minor_px):
        y1 = min(y + minor_thin, height)
        img[y:y1, :] = minor_color

    # Every 10 mm: vivid horizontal + vertical lines
    vivid = [
        (0, 0, 255),      # red
        (0, 200, 0),      # green
        (255, 0, 0),      # blue
        (255, 0, 255),    # magenta
        (200, 0, 200),    # purple
        (0, 128, 255),    # orange
        (255, 255, 0),    # cyan
        (0, 0, 128),      # dark red
    ]
    major_x = list(range(0, width, major_px))
    major_y = list(range(0, height, major_px))
    for i, x in enumerate(major_x):
        color = vivid[i % len(vivid)]
        x1 = min(x + major_thick, width)
        img[:, x:x1] = color
    for i, y in enumerate(major_y):
        color = vivid[(i + 3) % len(vivid)]
        y1 = min(y + major_thick, height)
        img[y:y1, :] = color

    # Origin marker at (0, 0)
    mark = min(major_px // 2, 40)
    img[0:mark, 0:mark] = (0, 0, 0)
    cx = mark // 2
    img[cx:cx + 1, 0:mark] = (255, 255, 255)
    img[0:mark, cx:cx + 1] = (255, 255, 255)

    # Label every 10 mm block: mm coordinates
    font = cv2.FONT_HERSHEY_SIMPLEX
    for j, y in enumerate(major_y):
        for i, x in enumerate(major_x):
            mm_x = i * 10
            mm_y = j * 10
            label = f"{mm_x},{mm_y}mm"
            tx, ty = x + 4, y + 16
            if tx + 60 < width and ty + 4 < height:
                cv2.putText(img, label, (tx, ty), font, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    return img


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate 1 mm grid with 10 mm vivid lines.")
    parser.add_argument("-o", "--output", type=Path, default=Path("../image/image.png"))
    parser.add_argument("--width", type=int, default=2560)
    parser.add_argument("--height", type=int, default=1440)
    parser.add_argument("--dpi", type=float, default=254.0, help="254 DPI => 1 mm = 10 px")
    parser.add_argument("--minor-mm", type=float, default=1.0, help="fine grid spacing in mm")
    parser.add_argument("--major-mm", type=float, default=10.0, help="vivid line spacing in mm")
    parser.add_argument("--minor-px", type=int, default=0, help="override 1 mm size in pixels")
    parser.add_argument("--major-px", type=int, default=0, help="override 10 mm size in pixels")
    args = parser.parse_args()

    minor_px = args.minor_px if args.minor_px > 0 else mm_to_px(args.minor_mm, args.dpi)
    major_px = args.major_px if args.major_px > 0 else mm_to_px(args.major_mm, args.dpi)

    img = build_pattern(args.width, args.height, minor_px, major_px)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), img):
        raise RuntimeError(f"failed to write {args.output}")

    print(f"written: {args.output}")
    print(f"size: {args.width}x{args.height}")
    print(f"1 mm = {minor_px} px, 10 mm = {major_px} px (dpi={args.dpi})")
    print(f"fine grid: {args.width // minor_px} x {args.height // minor_px} cells")
    print(f"major lines: {args.width // major_px + 1} vertical, {args.height // major_px + 1} horizontal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
