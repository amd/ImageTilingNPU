@echo off
REM Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
REM Licensed under the MIT License.
setlocal
cd /d %~dp0..

cmake -S . -B build
if errorlevel 1 exit /b 1

cmake --build build --config Release
if errorlevel 1 exit /b 1

echo Binaries copied to rel\ by CMake POST_BUILD.
if exist rel\*.exe (
    dir /b rel\*.exe
) else (
    echo WARNING: no executables found under rel\
)
pause
