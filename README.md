# ImageTilingNPU

## Project introduction

This repository provides an implementation for deploying neural network-based image super-resolution models on the AMD XDNA™ platform, leveraging the NPU (Neural Processing Unit) for efficient inference. The goal is to enable fast, low-power, and high-quality super-resolution on edge and embedded devices powered by AMD XDNA™.

## Features

- High-quality super-resolution with tiled inference on large images
- AMD XDNA NPU acceleration via VitisAI EP (ONNX) or graph-engine (xmodel)
- **ONNX** and **xmodel** model support
- Functional and performance test executables
- CMake POST_BUILD copies binaries and runtime DLLs into `rel/`

## Requirements

- Windows with AMD XDNA™ NPU hardware and driver installed
- Visual Studio 2022
- CMake ≥ 3.18
- Python 3

## Quick Start

### 1. Prepare dependencies

Download [imagetiling_dep.zip](https://download.amd.com/opendownload/RyzenAI/NPU_image_tiling/imagetiling_dep.zip), extract it, and copy **`voe_package`** and **`third_party`** to the project root:

```
ImageTilingNPU/
├── voe_package/
├── third_party/
├── scripts/
├── src/
└── ...
```

Place your ONNX model at `model/test.onnx` and xclbin at `xclbin/` (paths used by the scripts below).

### 2. Run scripts in order

From the project root, execute the following scripts under `scripts/` **in this order**:

| Step | Script                           | Purpose                                                               |
| ---- | -------------------------------- | --------------------------------------------------------------------- |
| 1    | `scripts\build.bat`              | Configure and build Release; binaries are copied to `rel/`            |
| 2    | `scripts\inject_vaip_stride.bat` | Inject SR stride shapes into VAIP config from ONNX I/O and image size |
| 3    | `scripts\gen_xmodel.bat`         | Compile ONNX → xmodel with VitisAI EP                                 |
| 4    | `scripts\gen_offset_grid.bat`    | Generate offset grid for the input image                              |
| 5    | `scripts\func_xmodel.bat`        | Functional test with zero-copy xmodel                                 |
| 6    | `scripts\func_onnx.bat`          | Functional test with ONNX (CPU EP by default)                         |
| 7    | `scripts\perf_xmodel.bat`        | xmodel performance test (sets NPU turbo mode, runs 60 s)              |

```powershell
scripts\build.bat
scripts\inject_vaip_stride.bat
scripts\gen_xmodel.bat
scripts\gen_offset_grid.bat
scripts\func_xmodel.bat
scripts\func_onnx.bat
scripts\perf_xmodel.bat
```

After `build.bat`, `rel/` contains:

- `test_SR_onnx.exe`, `perf_SR_onnx.exe`, `test_SR_xmodel.exe`, `perf_SR_xmodel.exe`
- `vitis-ai-runtime.dll`, `opencv_world4110.dll`
- ONNX Runtime DLLs (from `voe_package/bin/`)

Add XRT `bin` to `PATH` when running xmodel tests if `xrt_coreutil.dll` is not in `rel/`.

## Repository layout

| Path           | Description                                                                 |
| -------------- | --------------------------------------------------------------------------- |
| `src/`         | Application and runner source code                                          |
| `scripts/`     | Build, test, and model-preparation helper scripts                           |
| `voe_package/` | ONNX Runtime + VitisAI EP headers/libs/DLLs (from imagetiling_dep.zip)      |
| `third_party/` | vitis-ai-runtime, OpenCV, glog, XRT headers/libs (from imagetiling_dep.zip) |
| `vaip_config/` | Generated VAIP config with stride shapes                                    |
| `model/`       | ONNX / xmodel files and compile cache                                       |
| `image/`       | Sample input images                                                         |
| `xclbin/`      | NPU overlay xclbin                                                          |
| `rel/`         | Runtime output folder (exe + DLLs, created at build time)                   |
| `build/`       | CMake build directory                                                       |

## Executables

| Binary               | Model     | Description                                             |
| -------------------- | --------- | ------------------------------------------------------- |
| `test_SR_onnx.exe`   | `.onnx`   | Functional test, sol3-aligned tiling + quant pre/post   |
| `perf_SR_onnx.exe`   | `.onnx`   | Performance test (FPS over `--perf_sec`)                |
| `test_SR_xmodel.exe` | `.xmodel` | Functional test, zero-copy stride xmodel (sol3-aligned) |
| `perf_SR_xmodel.exe` | `.xmodel` | Performance test for zero-copy xmodel                   |

## ONNX execution provider

For **`.onnx`** models in `test_SR_onnx` / `perf_SR_onnx`:

| Mode                 | Flags                                                        |
| -------------------- | ------------------------------------------------------------ |
| **CPU EP** (default) | Omit `--vitisai`                                             |
| **VitisAI EP**       | `--vitisai` plus `--vai_options "xclbin\|...\ ctx_idx\|..."` |

Shape probing always uses a CPU session; worker sessions use the selected EP.

### Quantization flags (ONNX)

| Flag                  | Default | Meaning                   |
| --------------------- | ------- | ------------------------- |
| `--input_zero_point`  | `0`     | Input uint8 → model input |
| `--input_y_scale`     | `256.0` | Input scale               |
| `--output_zero_point` | `0`     | Model output → uint8      |
| `--output_y_scale`    | `256.0` | Output scale              |

To run ONNX on NPU, uncomment the `--vitisai` line in `scripts\func_onnx.bat`.

## Manual command examples

Set `PATH` so `rel/` and `voe_package/bin` are visible:

```powershell
$env:PATH = ".\rel;.\voe_package\bin;$env:PATH"
```

**ONNX (CPU):**

```powershell
.\rel\test_SR_onnx.exe ^
  --model .\model\test.onnx ^
  --image .\image\image.png ^
  --output sr_onnx.png ^
  --threads 4
```

**ONNX (VitisAI NPU):**

```powershell
.\rel\test_SR_onnx.exe ^
  --model .\model\test.onnx ^
  --image .\image\image.png ^
  --output sr_onnx.png ^
  --threads 4 ^
  --vitisai ^
  --vai_options "xclbin|xclbin/AMD_AIE2P_8x4x1_Overlay.xclbin ctx_idx|1 target|X1-Benchmark"
```

**xmodel (zero-copy stride):**

```powershell
.\rel\test_SR_xmodel.exe ^
  --model .\model\test\compiled.xmodel ^
  --image .\image\image.png ^
  --output sr_xmodel.png ^
  --vai_options "xclbin|xclbin/AMD_AIE2P_8x4x1_Overlay.xclbin ctx_idx|1"
```

**xmodel performance:**

```powershell
xrt-smi configure --pmode turbo
.\rel\perf_SR_xmodel.exe ^
  --model .\model\test\compiled.xmodel ^
  --image_width 2560 --image_height 1440 ^
  --perf_sec 60 ^
  --vai_options "xclbin|xclbin/AMD_AIE2P_8x4x1_Overlay.xclbin ctx_idx|1"
```

## Contribute

If you encounter problems or want a new feature, please open an issue. Pull requests are welcome.
