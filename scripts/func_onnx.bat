@echo off
REM Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
REM Licensed under the MIT License.
cd /d %~dp0..
set ROOT=%cd%
set PATH=%ROOT%\rel;%ROOT%\voe_package\bin;%PATH%

.\rel\test_SR_onnx.exe ^
  --image .\image\image.png ^
  --model .\model\test.onnx ^
  --output sr_onnx.png ^
  --threads 4 ^
  --input_zero_point 0 ^
  --input_y_scale 256 ^
  --output_zero_point 0 ^
  --output_y_scale 256

REM For VitisAI EP, add:
REM   --vitisai ^
REM   --vai_options "xclbin|xclbin/AMD_AIE2P_8x4x1_Overlay.xclbin ctx_idx|1 target|X1-Benchmark"

pause
