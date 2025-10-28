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

#include "xmodel_runner.h"

namespace tiling_sr {

std::unique_ptr<XmodelRunner> XmodelRunner::create_instance(const xir::Graph *graph, const std::unordered_map<std::string, std::string>& options) {
	std::unique_ptr<XmodelRunner> ret = std::unique_ptr<XmodelRunner>(new XmodelRunner());
	if (ret->init(graph, options) != 0) {
		return nullptr;
	}
	return ret;
}

std::unique_ptr<XmodelRunner> XmodelRunner::create_instance(const std::string& model, const std::unordered_map<std::string, std::string>& options) {
	return create_instance(xir::Graph::deserialize(model).release(), options);
}

XmodelRunner::XmodelRunner() {

}

XmodelRunner::~XmodelRunner() {

}

int XmodelRunner::init(const xir::Graph *graph, const std::unordered_map<std::string, std::string>& options) {
	std::unique_ptr<xir::Attrs> attrs = xir::Attrs::create();
	if (options.find("ctx_idx") != options.end()) {
      	attrs->set_attr<int>("ctx_idx", atoi(options.at("ctx_idx").c_str()));
  	}
	if (options.find("xclbin") != options.end()) {
		std::string xclbin = options.at("xclbin");
    	attrs->set_attr<std::string>("xclbin_file", xclbin);
  	}

	if (options.find("work_mode") != options.end()) {
		std::string user_sub_bo = options.at("work_mode");
		attrs->set_attr<std::string>("work_mode", user_sub_bo);
	}

	try {
		_xmodel_runner = GraphEngine::create_graph_runner(graph, attrs.get());
    } catch (const std::exception& e) {
        LOG(ERROR) << "fail to load VitisAI : " << e.what();
        return -1;
    }

	// init io type
	for (vart::TensorBuffer* tb : _xmodel_runner->get_inputs()) {
		TensorInfo tensor_info;

		xir::DataType data_type = tb->get_tensor()->get_data_type();
		if (data_type.type == xir::DataType::XINT || data_type.type == xir::DataType::XUINT ||
			data_type.type == xir::DataType::UINT || data_type.type == xir::DataType::INT) {
			tensor_info.data_type = TensorInfo::DataType::INT8;
		} else {
			LOG(ERROR) << "unsupported input type : " << data_type.to_string();
			return -1;
		}

		std::vector<int32_t> shape = tb->get_tensor()->get_shape();
		for (int32_t s : shape) {
			tensor_info.shape.push_back(s);
		}

		_input_infos.push_back(tensor_info);
	}
	for (vart::TensorBuffer* tb : _xmodel_runner->get_outputs()) {
		TensorInfo tensor_info;

		xir::DataType data_type = tb->get_tensor()->get_data_type();
		if (data_type.type == xir::DataType::XINT || data_type.type == xir::DataType::XUINT ||
			data_type.type == xir::DataType::UINT || data_type.type == xir::DataType::INT) {
			tensor_info.data_type = TensorInfo::DataType::INT8;
		} else {
			LOG(ERROR) << "unsupported input type : " << data_type.to_string();
			return -1;
		}

		std::vector<int32_t> shape = tb->get_tensor()->get_shape();
		for (int32_t s : shape) {
			tensor_info.shape.push_back(s);
		}

		_output_infos.push_back(tensor_info);
	}

    return 0;
}

int XmodelRunner::get_runner_io(std::vector<TensorInfo>& input_infos, std::vector<TensorInfo>& output_infos) {
	input_infos = _input_infos;
	output_infos = _output_infos;
	return 0;
}

int XmodelRunner::run(std::vector<OrtBuf>& in_bufs, std::vector<OrtBuf>& out_bufs) {
	
    std::vector<vart::TensorBuffer*> inputs = _xmodel_runner->get_inputs();
    std::vector<vart::TensorBuffer*> outputs = _xmodel_runner->get_outputs();

	for (size_t i = 0; i < inputs.size(); ++i) {
		const auto [input_ptr, input_size] = inputs[i]->data();
		std::memcpy(reinterpret_cast<char*>(input_ptr), in_bufs[i].buf, input_size);
	}
	auto ret = _xmodel_runner->execute_async(inputs, outputs);
    _xmodel_runner->wait(ret.first, -1);
	for (size_t i = 0; i < outputs.size(); ++i) {
		const auto [output_ptr, output_size] = outputs[i]->data();
		std::memcpy(out_bufs[i].buf, reinterpret_cast<char*>(output_ptr), output_size);
	}

    return 0;
}

} // namespace tiling_sr