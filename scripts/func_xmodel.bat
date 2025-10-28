@echo off
REM Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
REM Licensed under the MIT License.
cd /d %~dp0..
set PATH=%cd%\rel;%PATH%

.\rel\test_SR_xmodel.exe ^
  --image .\image\image.png ^
  --vai_options "xclbin|xclbin/AMD_AIE2P_8x4x1_Overlay.xclbin ctx_idx|1" ^
  --model .\model\test\compiled.0xa000205003fdb46.xmodel ^
  --output sr_xmodel.png

pause
