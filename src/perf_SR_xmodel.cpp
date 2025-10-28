// Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:

//    http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <chrono>
#include <string>

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "xmodel_zero_copy.h"

DEFINE_string(model, "", "onnx model path");
DEFINE_string(vai_options, "", "");
DEFINE_int32(perf_sec, 60, "profiling seconds, -1 means no");
DEFINE_int32(image_width, 2560, "input image width");
DEFINE_int32(image_height, 1440, "input image height");

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    if (FLAGS_model == "") {
        LOG(ERROR) << "invalid model";
        return -1;
    }

    auto session = tiling_sr::XmodelZeroCopySession::open(
        FLAGS_model, tiling_sr::parse_vai_options(FLAGS_vai_options),
        FLAGS_image_width, FLAGS_image_height);
    if (session == nullptr) {
        return -1;
    }
    if (!session->create_bos()) {
        return -1;
    }

    session->add_to_runlist();

    std::chrono::high_resolution_clock::time_point begin =
        std::chrono::high_resolution_clock::now();
    size_t count = 0;
    LOG(INFO) << "start to profile";
    while (1) {
        session->input_sync();
        if (std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - begin).count() / 1000 >=
            FLAGS_perf_sec * 1000) {
            break;
        }
        session->execute_runlist();
        session->output_sync();
        ++count;
    }

    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - begin).count() / 1000.0 / 1000.0;

    LOG(INFO) << "Processed " << count << " frames in " << elapsed << " seconds";
    LOG(INFO) << "Average FPS: " << count / elapsed;
    return 0;
}
