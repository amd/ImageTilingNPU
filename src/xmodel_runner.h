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

#ifndef _XMODLE_RUNNER_H_
#define _XMODEL_RUNNER_H_

#include <memory>
#include <string>
#include <unordered_map>

#include "graph-engine/create_graph_runner.hpp"
#include "vart/runner_ext.hpp"

#include "base_runner.h"

namespace tiling_sr {

class XmodelRunner : public BaseRunner {
public:
    static std::unique_ptr<XmodelRunner> create_instance(const std::string& model, const std::unordered_map<std::string, std::string>& options);
    static std::unique_ptr<XmodelRunner> create_instance(const xir::Graph *graph, const std::unordered_map<std::string, std::string>& options);
    ~XmodelRunner();
    int get_runner_io(std::vector<TensorInfo>& input_infos, std::vector<TensorInfo>& output_infos);
    int run(std::vector<OrtBuf>& input, std::vector<OrtBuf>& output);
    xrt::bo* create_parent_bo_input(std::vector<void*>& input_ptrs, std::vector<size_t> input_size) {
        return _xmodel_runner->create_parent_bo_input(input_ptrs, input_size);
    }
    xrt::bo* create_sub_bo_input(xrt::bo* parent_bo, std::size_t size, std::size_t offset) {
        return _xmodel_runner->create_sub_bo_input(parent_bo, size, offset);
    }
    xrt::bo* create_parent_bo_output(std::vector<void*>& output_ptrs, std::vector<size_t> output_size) {
        return _xmodel_runner->create_parent_bo_output(output_ptrs, output_size);
    }
    xrt::bo* create_sub_bo_output(xrt::bo* parent_bo, std::size_t size, std::size_t offset) {
        return _xmodel_runner->create_sub_bo_output(parent_bo, size, offset);
    }
    void input_sync(xrt::bo* input, xrt::bo* output) {
        _xmodel_runner->input_sync(input, output);
    }
    void output_sync(xrt::bo* output) {
        _xmodel_runner->output_sync(output);
    }
    void add_to_runlist(uint32_t index) {
        _xmodel_runner->add_to_runlist(_inputs, _outputs, index);
    }
    void execute_runlist(uint32_t index) {
        _xmodel_runner->execute_runlist(index);
    }
private:
    XmodelRunner();
    int init(const xir::Graph *graph, const std::unordered_map<std::string, std::string>& options);
private:
    std::unique_ptr<vart::RunnerExt> _xmodel_runner;
    std::vector<TensorInfo> _input_infos;
    std::vector<TensorInfo> _output_infos;
    std::vector<vart::TensorBuffer*> _inputs;
    std::vector<vart::TensorBuffer*> _outputs;
};

} // namespace tiling_sr

#endif // _XMODLE_RUNNER_H_