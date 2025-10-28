@echo off
REM Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
REM Licensed under the MIT License.
setlocal
cd /d %~dp0..

set ROOT=%cd%
set PATH=%ROOT%\voe_package\bin;%PATH%

set MODEL=model\test.onnx
set MODEL_DIR=%ROOT%\model
set CONFIG=%ROOT%\vaip_config\vaip_config_stride.json
set CACHE_KEY=test

if not exist "%ROOT%\voe_package\bin\onnxruntime_perf_test.exe" (
    echo error: onnxruntime_perf_test.exe not found in voe_package\bin
    exit /b 1
)
if not exist "%ROOT%\%MODEL%" (
    echo error: model not found: %ROOT%\%MODEL%
    exit /b 1
)
if not exist "%CONFIG%" (
    echo error: config not found: %CONFIG%
    echo run scripts\inject_vaip_stride.bat first
    exit /b 1
)

echo ROOT      : %ROOT%
echo MODEL     : %MODEL%
echo MODEL_DIR : %MODEL_DIR%
echo CONFIG    : %CONFIG%
echo CACHE_KEY : %CACHE_KEY%
echo.

if exist "%MODEL_DIR%\%CACHE_KEY%" (
    echo clearing cache: %MODEL_DIR%\%CACHE_KEY%
    rmdir /s /q "%MODEL_DIR%\%CACHE_KEY%"
)

onnxruntime_perf_test.exe -I -t 3 -e vitisai -i "config_file|%CONFIG% enable_cache_file_io_in_mem|0 target|X1-Benchmark cache_dir|%MODEL_DIR% cache_key|%CACHE_KEY%" -c 1 "%ROOT%\%MODEL%"

pause
