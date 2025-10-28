# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.18.1)

project(ortgenaiapp)
set(CMAKE_CXX_STANDARD 20)
add_compile_definitions(GLOG_NO_ABBREVIATED_SEVERITIES)

add_compile_options(
    /Zc:__cplusplus
    /Zi
    /ZH:SHA_256
    /guard:cf
    /sdl
    /MP
)

set(ORT_DIR voe_package)
set(ORT_LIB_DIR ${ORT_DIR}/lib)
set(ORT_INC_DIR ${ORT_DIR}/include/onnxruntime/)
set(VOE_BIN_DIR ${CMAKE_SOURCE_DIR}/${ORT_DIR}/bin)
set(THIRD_PARTY_DIR ${CMAKE_SOURCE_DIR}/third_party)
set(REL_DIR ${CMAKE_SOURCE_DIR}/rel)
set(VITIS_AI_RUNTIME_DLL ${THIRD_PARTY_DIR}/bin/vitis-ai-runtime.dll)
set(OPENCV_DLL ${THIRD_PARTY_DIR}/bin/opencv_world4110.dll)
set(ORT_RUNTIME_DLLS
    onnxruntime.dll
    onnxruntime_providers_shared.dll
    onnxruntime_providers_vitisai.dll
    onnxruntime_vitisai_ep.dll
    onnxruntime_vitis_ai_custom_ops.dll
)

add_executable(perf_SR_onnx
    src/perf_SR_onnx.cpp
    src/onnx_runner.cpp
)
target_link_directories(perf_SR_onnx PRIVATE ${ORT_LIB_DIR} third_party/lib/)
target_include_directories(perf_SR_onnx PRIVATE ${ORT_INC_DIR} third_party/include/)
target_link_libraries(perf_SR_onnx PUBLIC onnxruntime glog gflags_static Shlwapi opencv_world4110 vitis-ai-runtime)

add_executable(test_SR_onnx
    src/test_SR_onnx.cpp
    src/onnx_runner.cpp
)
target_link_directories(test_SR_onnx PRIVATE ${ORT_LIB_DIR} third_party/lib/)
target_include_directories(test_SR_onnx PRIVATE ${ORT_INC_DIR} third_party/include/)
target_link_libraries(test_SR_onnx PUBLIC onnxruntime glog gflags_static Shlwapi opencv_world4110 vitis-ai-runtime)

add_executable(perf_SR_xmodel
    src/perf_SR_xmodel.cpp
    src/xmodel_runner.cpp
    src/xmodel_zero_copy.cpp
)
target_link_directories(perf_SR_xmodel PRIVATE ${ORT_LIB_DIR} third_party/lib/)
target_include_directories(perf_SR_xmodel PRIVATE ${ORT_INC_DIR} third_party/include/ third_party/include/xrt/)
target_link_libraries(perf_SR_xmodel PUBLIC glog gflags_static Shlwapi vitis-ai-runtime xrt_coreutil Cfgmgr32)

add_executable(test_SR_xmodel
    src/test_SR_xmodel.cpp
    src/xmodel_runner.cpp
    src/xmodel_zero_copy.cpp
)
target_link_directories(test_SR_xmodel PRIVATE ${ORT_LIB_DIR} third_party/lib/)
target_include_directories(test_SR_xmodel PRIVATE ${ORT_INC_DIR} third_party/include/ third_party/include/xrt/)
target_link_libraries(test_SR_xmodel PUBLIC glog gflags_static Shlwapi opencv_world4110 vitis-ai-runtime xrt_coreutil)

foreach(target perf_SR_onnx test_SR_onnx perf_SR_xmodel test_SR_xmodel)
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${REL_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:${target}>
            ${REL_DIR}/$<TARGET_FILE_NAME:${target}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${VITIS_AI_RUNTIME_DLL}
            ${REL_DIR}/vitis-ai-runtime.dll
        COMMENT "Copy ${target} and vitis-ai-runtime.dll to rel/"
    )
endforeach()

foreach(target perf_SR_onnx test_SR_onnx perf_SR_xmodel test_SR_xmodel)
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${OPENCV_DLL}
            ${REL_DIR}/opencv_world4110.dll
        COMMENT "Copy opencv_world4110.dll to rel/ for ${target}"
    )
endforeach()

foreach(target perf_SR_onnx test_SR_onnx)
    foreach(ort_dll ${ORT_RUNTIME_DLLS})
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${VOE_BIN_DIR}/${ort_dll}
                ${REL_DIR}/${ort_dll}
            COMMENT "Copy ${ort_dll} to rel/ for ${target}"
        )
    endforeach()
endforeach()
