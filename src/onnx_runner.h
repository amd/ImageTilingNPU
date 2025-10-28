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

#ifndef _ONNX_RUNNER_H_
#define _ONNX_RUNNER_H_

#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "base_runner.h"

namespace tiling_sr {
	
class ONNXRunner : public BaseRunner {
public:
	static std::unique_ptr<ONNXRunner> create_instance(const std::string& model, bool npu, const std::unordered_map<std::string, std::string>& options);
	~ONNXRunner();
    int get_runner_io(std::vector<TensorInfo>& input_infos, std::vector<TensorInfo>& output_infos);
    int run(std::vector<OrtBuf>& input, std::vector<OrtBuf>& output);
private:
	int init(const std::string& model, bool npu, const std::unordered_map<std::string, std::string>& options);
	ONNXRunner();
private:
	Ort::AllocatorWithDefaultOptions _allocator;
    Ort::Env _env;
    std::unique_ptr<Ort::Session> _session_ptr;

    std::vector<const char*> _input_names;
    std::vector<Ort::AllocatedStringPtr> _input_names_ptr;
    std::vector<std::vector<int64_t>> _input_shapes;
    std::vector<TensorInfo> _input_infos;

    std::vector<const char*> _output_names;
    std::vector<Ort::AllocatedStringPtr> _output_names_ptr;
    std::vector<std::vector<int64_t>> _output_shapes;
    std::vector<TensorInfo> _output_infos;
};
	
} // namespace gaiming_sr

#endif // _ONNX_RUNNER_H_