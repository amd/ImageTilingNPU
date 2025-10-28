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

#include "onnx_runner.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <fstream>
#include <random>
#include <windows.h>

#include "glog/logging.h"

namespace tiling_sr {

std::wstring string_to_wstring(const std::string& str) {
    if (str.empty()) {
        return L"";
    }

    const auto size_needed = MultiByteToWideChar(
        CP_UTF8, 0, str.data(), static_cast<int>(str.size()), nullptr, 0);
    if (size_needed <= 0) {
        throw std::runtime_error("MultiByteToWideChar() failed: " + std::to_string(size_needed));
    }

    std::wstring result(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.data(), static_cast<int>(str.size()), result.data(), size_needed);
    return result;
}
	
ONNXRunner::ONNXRunner()  :
    _env(ORT_LOGGING_LEVEL_WARNING, "onnx_runner") {
	
}

ONNXRunner::~ONNXRunner() {
	
}

std::unique_ptr<ONNXRunner> ONNXRunner::create_instance(const std::string& model, bool npu, const std::unordered_map<std::string, std::string>& options) {
	std::unique_ptr<ONNXRunner> ret = std::unique_ptr<ONNXRunner>(new ONNXRunner());
	if (ret->init(model, npu, options) != 0) {
		return nullptr;
	}
	return ret;
}

int ONNXRunner::run(std::vector<OrtBuf>& input_bufs, std::vector<OrtBuf>& output_bufs) {
    
    Ort::IoBinding io_binding(*_session_ptr);

    for (size_t i = 0; i < _session_ptr->GetInputCount(); i++) {
        auto input_type = _session_ptr->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType();
        
        Ort::MemoryInfo info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value input_tensor = Ort::Value::CreateTensor(info, input_bufs[i].buf, input_bufs[i].len,
            _input_shapes[i].data(), _input_shapes[i].size(), input_type);
        
        io_binding.BindInput(_input_names[i], input_tensor);
    }

    for (size_t i = 0; i < _session_ptr->GetOutputCount(); i++) {
        auto output_type = _session_ptr->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType();
        
        Ort::MemoryInfo info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value output_tensor = Ort::Value::CreateTensor(info, output_bufs[i].buf, output_bufs[i].len,
            _output_shapes[i].data(), _output_shapes[i].size(), output_type);

        io_binding.BindOutput(_output_names[i], output_tensor);
    }
    try {
        _session_ptr->Run(Ort::RunOptions(), io_binding);
    } catch (const std::exception& e) {
        LOG(ERROR) << "error : " << e.what();
        return -1;
    }

    return 0;
}

int ONNXRunner::get_runner_io(std::vector<TensorInfo>& input_infos, std::vector<TensorInfo>& output_infos) {
    input_infos = _input_infos;
    output_infos = _output_infos;
    return 0;
}

int ONNXRunner::init(const std::string& model, bool npu, const std::unordered_map<std::string, std::string>& options) {
    Ort::SessionOptions session_options;
    session_options.AddConfigEntry("session.intra_op.allow_spinning", "0");
    if (npu) {
        try {
        session_options.AppendExecutionProvider_VitisAI(options);
        }catch (const std::exception& e) {
            LOG(ERROR) << "fail to load VitisAI : " << e.what();
            return -1;
        }
    }
    
	try {
        _session_ptr = std::make_unique<Ort::Session>(_env, string_to_wstring(model).data(), session_options);
    } catch (const std::exception& e) {
        LOG(ERROR) << "fail to create onnx runner session" << e.what();
        return -1;
    } 

    for (size_t i = 0; i < _session_ptr->GetInputCount(); i++) {
        auto name = _session_ptr->GetInputNameAllocated(i, _allocator);        
        _input_names.push_back(name.get());
        _input_names_ptr.push_back(std::move(name));
        
        TensorInfo tensor_info;
        auto input_shape = _session_ptr->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        tensor_info.shape = input_shape;
        _input_shapes.push_back(input_shape);

        auto input_type = _session_ptr->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType();
        if (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            tensor_info.data_type = TensorInfo::DataType::FLOAT32;
        } else if (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8) {
            tensor_info.data_type = TensorInfo::DataType::INT8;
        } else if (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
            tensor_info.data_type = TensorInfo::DataType::UINT8;
        } else {
            LOG(ERROR) << "unsupported output type : " << input_type;
            return -1;
        }
        _input_infos.push_back(tensor_info);
    }
    for (size_t i = 0; i < _session_ptr->GetOutputCount(); i++) {
        auto name = _session_ptr->GetOutputNameAllocated(i, _allocator);        
        _output_names.push_back(name.get());
        _output_names_ptr.push_back(std::move(name));
        
        TensorInfo tensor_info;
        auto output_shape = _session_ptr->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        tensor_info.shape = output_shape;
        _output_shapes.push_back(output_shape);

        auto output_type = _session_ptr->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType();
        if (output_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            tensor_info.data_type = TensorInfo::DataType::FLOAT32;
        } else if (output_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8) {
            tensor_info.data_type = TensorInfo::DataType::INT8;
        } else if (output_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
            tensor_info.data_type = TensorInfo::DataType::UINT8;
        } else {
            LOG(ERROR) << "unsupported output type : " << output_type;
            return -1;
        }
        _output_infos.push_back(tensor_info);
    }

	return 0;
}
	
} // namespace tiling_sr