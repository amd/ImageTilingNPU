@echo off
REM Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
REM Licensed under the MIT License.
cd /d %~dp0..

python scripts\gen_offset_grid.py -o image\image.png --width 2560 --height 1440
pause
