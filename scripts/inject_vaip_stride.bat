@echo off
REM Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
REM Licensed under the MIT License.
cd /d %~dp0..
py -3 scripts\inject_vaip_stride.py ^
  --model model\test.onnx ^
  --image-width 2560 ^
  --image-height 1440 ^
  --target X1-Benchmark
pause
