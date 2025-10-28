@echo off
REM Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
REM Licensed under the MIT License.
cd /d %~dp0..
set PATH=%cd%\rel;%PATH%
C:\WINDOWS\System32\AMD\xrt-smi.exe configure --pmode turbo
.\rel\perf_SR_xmodel.exe ^
  --image_width 2560 ^
  --image_height 1440 ^
  --perf_sec 60 ^
  --vai_options "xclbin|xclbin/AMD_AIE2P_8x4x1_Overlay.xclbin ctx_idx|1" ^
  --model .\model\test\compiled.0xa000205003fdb46.xmodel

pause
