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

#ifndef _BASE_RUNNER_H_
#define _BASE_RUNNER_H_

#include <vector>

namespace tiling_sr {

struct OrtBuf {
    void* buf;
    size_t len;
};

struct TensorInfo {
    std::vector<int64_t> shape;
    enum DataType {
        FLOAT32,
        UINT8,
        INT8
    };
    DataType data_type;
    size_t shape_size() {
        int64_t ret = 1;
        for (int64_t s : shape) {
            ret *= s;
        }
        return ret;
    }
    size_t ele_size() {
        size_t ele_size = 1;
        if (data_type == DataType::FLOAT32) {
            ele_size = 4;
        } else {
            ele_size = 1;
        }
        return ele_size;
    }
    size_t tensor_size() {
        return ele_size() * shape_size();
    }
};

class BaseRunner {
public:
    BaseRunner() {

    }
    ~BaseRunner() {

    }
    virtual int get_runner_io(std::vector<TensorInfo>& input_infos, std::vector<TensorInfo>& output_infos) = 0;
    virtual int run(std::vector<OrtBuf>& input, std::vector<OrtBuf>& output) {
        return 0;
    }
    virtual int run() {
        return 0;
    }
    virtual void set_buffers(std::vector<OrtBuf>& input, std::vector<OrtBuf>& output){}

    virtual int32_t get_input_size() {
        return 0;
    }
    virtual int32_t get_output_size() {
        return 0;
    }
};

} // namespace tiling_sr

#endif //_BASE_RUNNER_H_