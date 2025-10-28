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

#include "xmodel_zero_copy.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <sstream>
#include <vector>

#include "glog/logging.h"

namespace tiling_sr {
namespace {

const xir::Subgraph* find_dpu_subgraph(const xir::Subgraph* subgraph) {
    if (subgraph == nullptr) {
        return nullptr;
    }
    if (subgraph->has_attr("device") &&
        subgraph->get_attr<std::string>("device") == "DPU") {
        return subgraph;
    }
    for (const xir::Subgraph* child : subgraph->children_topological_sort()) {
        const xir::Subgraph* match = find_dpu_subgraph(child);
        if (match != nullptr) {
            return match;
        }
    }
    return nullptr;
}

std::string vec_to_string(const std::vector<int>& values) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            oss << ",";
        }
        oss << values[i];
    }
    oss << "]";
    return oss.str();
}

std::vector<int> to_int_vec(const std::vector<int32_t>& values) {
    return std::vector<int>(values.begin(), values.end());
}

std::vector<int> compact_element_stride(const std::vector<int>& shape) {
    std::vector<int> stride(shape.size());
    int step = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        stride[i] = step;
        step *= shape[i];
    }
    return stride;
}

std::vector<int> tensor_align_shape(const xir::Tensor* tensor) {
    if (tensor->has_attr("align_shape")) {
        return tensor->get_attr<std::vector<int> >("align_shape");
    }
    return to_int_vec(tensor->get_shape());
}

bool tensor_needs_pad(const xir::Tensor* tensor) {
    if (tensor == nullptr || !tensor->has_attr("stride")) {
        return false;
    }
    const auto align_shape = tensor_align_shape(tensor);
    const auto stride = tensor->get_attr<std::vector<int> >("stride");
    if (stride.size() != align_shape.size()) {
        return true;
    }
    if (stride == align_shape) {
        return false;
    }
    return stride != compact_element_stride(align_shape);
}

std::string tensor_stride_to_string(const xir::Tensor* tensor) {
    if (tensor == nullptr || !tensor->has_attr("stride")) {
        return "none";
    }
    return vec_to_string(tensor->get_attr<std::vector<int> >("stride"));
}

const xir::Op* find_upload_for_dpu_input(const xir::Subgraph* dpu_subgraph,
    const xir::Tensor* dpu_input) {
    const xir::Op* upload = nullptr;
    for (const xir::Op* op : dpu_subgraph->get_ops()) {
        if (op->get_type() != "upload") {
            continue;
        }
        bool consumes_input = false;
        for (const xir::Tensor* tensor : op->get_input_tensors()) {
            if (tensor == dpu_input) {
                consumes_input = true;
                break;
            }
        }
        if (!consumes_input) {
            continue;
        }
        if (upload != nullptr) {
            return nullptr;
        }
        upload = op;
    }
    return upload;
}

size_t tensor_ele_size(const xir::Tensor* tensor) {
    const int bit_width = tensor->get_data_type().bit_width;
    if (bit_width <= 0) {
        return 1;
    }
    return static_cast<size_t>((bit_width + 7) / 8);
}

bool read_stride_hw(const xir::Subgraph* subgraph, const char* key,
    size_t* height, size_t* width) {
    if (subgraph == nullptr || !subgraph->has_attr(key)) {
        LOG(ERROR) << "root subgraph missing attribute " << key;
        return false;
    }
    const auto shape = subgraph->get_attr<std::vector<int> >(key);
    if (shape.size() != 4 || shape[0] != 1 || shape[1] <= 0 || shape[2] <= 0) {
        LOG(ERROR) << "invalid " << key << ", expected [1,H,W,C], got "
            << vec_to_string(shape);
        return false;
    }
    *height = static_cast<size_t>(shape[1]);
    *width = static_cast<size_t>(shape[2]);
    return true;
}

size_t align_up(size_t value, size_t align) {
    return ((value + align - 1) / align) * align;
}

} // namespace

std::unordered_map<std::string, std::string> parse_vai_options(
    const std::string& vai_options) {
    std::unordered_map<std::string, std::string> options;
    std::istringstream iss(vai_options);
    std::string token;
    while (iss >> token) {
        size_t delimiter_pos = token.find('|');
        if (delimiter_pos != std::string::npos) {
            std::string key = token.substr(0, delimiter_pos);
            std::string value = token.substr(delimiter_pos + 1);
            options[key] = value;
            LOG(INFO) << "key : " << key << ", value : " << value;
        }
    }
    return options;
}

XmodelZeroCopySession::~XmodelZeroCopySession() {
    if (input_parent_buf_ != nullptr) {
        _aligned_free(input_parent_buf_);
    }
    if (output_parent_buf_ != nullptr) {
        _aligned_free(output_parent_buf_);
    }
}

std::unique_ptr<XmodelZeroCopySession> XmodelZeroCopySession::open(
    const std::string& model,
    const std::unordered_map<std::string, std::string>& options,
    int image_width, int image_height) {
    std::unique_ptr<XmodelZeroCopySession> session(new XmodelZeroCopySession());
    if (session->init(model, options, image_width, image_height) != 0) {
        return nullptr;
    }
    return session;
}

int XmodelZeroCopySession::init(const std::string& model,
    const std::unordered_map<std::string, std::string>& options,
    int image_width, int image_height) {
    graph_ = xir::Graph::deserialize(model);
    if (graph_ == nullptr) {
        LOG(ERROR) << "fail to deserialize xmodel : " << model;
        return -1;
    }

    const xir::Subgraph* dpu_subgraph =
        find_dpu_subgraph(graph_->get_root_subgraph());
    if (dpu_subgraph == nullptr) {
        LOG(ERROR) << "compiled xmodel does not contain a DPU subgraph";
        return -1;
    }
    const auto dpu_inputs = dpu_subgraph->get_sorted_input_tensors();
    const auto dpu_outputs = dpu_subgraph->get_sorted_output_tensors();
    if (dpu_inputs.size() != 1 || dpu_outputs.size() != 1) {
        LOG(ERROR) << "DPU subgraph must have exactly one input and one output"
            << ", inputs : " << dpu_inputs.size()
            << ", outputs : " << dpu_outputs.size();
        return -1;
    }

    const xir::Op* upload_op =
        find_upload_for_dpu_input(dpu_subgraph, dpu_inputs[0]);
    if (upload_op == nullptr || upload_op->get_type() != "upload") {
        LOG(ERROR) << "DPU subgraph input tensor must correspond to an upload op";
        return -1;
    }
    if (upload_op->get_output_tensor() == nullptr) {
        LOG(ERROR) << "upload op must have exactly one output tensor";
        return -1;
    }
    const xir::Tensor* upload_output_tensor = upload_op->get_output_tensor();

    const xir::Op* download_op = dpu_outputs[0]->get_producer();
    if (download_op == nullptr || download_op->get_type() != "download") {
        LOG(ERROR) << "DPU subgraph output tensor op type must be download";
        return -1;
    }
    const auto download_input_tensors = download_op->get_input_tensors();
    if (download_input_tensors.size() != 1) {
        LOG(ERROR) << "download op must have exactly one input tensor, got "
            << download_input_tensors.size();
        return -1;
    }
    const xir::Tensor* download_input_tensor = download_input_tensors[0];

    if (tensor_needs_pad(upload_output_tensor)) {
        LOG(ERROR) << "zero-copy cannot have pad"
            << ", upload output shape="
            << vec_to_string(to_int_vec(upload_output_tensor->get_shape()))
            << ", align_shape="
            << vec_to_string(tensor_align_shape(upload_output_tensor))
            << ", stride=" << tensor_stride_to_string(upload_output_tensor);
        return -1;
    }
    if (tensor_needs_pad(download_input_tensor)) {
        LOG(ERROR) << "zero-copy cannot have depad"
            << ", download input shape="
            << vec_to_string(to_int_vec(download_input_tensor->get_shape()))
            << ", align_shape="
            << vec_to_string(tensor_align_shape(download_input_tensor))
            << ", stride=" << tensor_stride_to_string(download_input_tensor);
        return -1;
    }

    if (!dpu_subgraph->has_attr("reg_id_to_size") ||
        !upload_output_tensor->has_attr("reg_id") ||
        !upload_output_tensor->has_attr("ddr_addr") ||
        !download_input_tensor->has_attr("reg_id") ||
        !download_input_tensor->has_attr("ddr_addr")) {
        LOG(ERROR) << "compiled DPU interface is missing register placement "
            "metadata";
        return -1;
    }

    const auto reg_id_to_size =
        dpu_subgraph->get_attr<std::map<std::string, int> >("reg_id_to_size");
    const int input_reg_id = upload_output_tensor->get_attr<int>("reg_id");
    const int output_reg_id = download_input_tensor->get_attr<int>("reg_id");
    const int input_ddr_addr_i = upload_output_tensor->get_attr<int>("ddr_addr");
    const int output_ddr_addr_i = download_input_tensor->get_attr<int>("ddr_addr");
    const std::string input_reg_name = "REG_" + std::to_string(input_reg_id);
    const std::string output_reg_name = "REG_" + std::to_string(output_reg_id);
    const auto input_reg_size_it = reg_id_to_size.find(input_reg_name);
    const auto output_reg_size_it = reg_id_to_size.find(output_reg_name);
    if (input_ddr_addr_i < 0 || output_ddr_addr_i < 0 ||
        input_reg_size_it == reg_id_to_size.end() ||
        output_reg_size_it == reg_id_to_size.end() ||
        input_reg_size_it->second <= 0 || output_reg_size_it->second <= 0) {
        LOG(ERROR) << "invalid compiled DPU register placement"
            << ", input=" << input_reg_name << "+" << input_ddr_addr_i
            << ", output=" << output_reg_name << "+" << output_ddr_addr_i;
        return -1;
    }
    input_ddr_addr_ = static_cast<size_t>(input_ddr_addr_i);
    output_ddr_addr_ = static_cast<size_t>(output_ddr_addr_i);
    input_reg_size_ = static_cast<size_t>(input_reg_size_it->second);
    output_reg_size_ = static_cast<size_t>(output_reg_size_it->second);

    const auto input_shape32 = upload_output_tensor->get_shape();
    const auto output_shape32 = download_input_tensor->get_shape();
    input_ele_size_ = tensor_ele_size(upload_output_tensor);
    output_ele_size_ = tensor_ele_size(download_input_tensor);
    if (input_shape32.size() != 4 || output_shape32.size() != 4) {
        LOG(ERROR) << "DPU input/output tensors must be rank-4"
            << ", input rank : " << input_shape32.size()
            << ", output rank : " << output_shape32.size();
        return -1;
    }

    const bool model_in_hwc = (input_shape32[3] == 4 || input_shape32[3] == 3);
    input_h_ = model_in_hwc ? input_shape32[1] : input_shape32[2];
    input_w_ = model_in_hwc ? input_shape32[2] : input_shape32[3];
    input_c_ = model_in_hwc ? input_shape32[3] : input_shape32[1];
    const bool model_out_hwc = (output_shape32[3] == 4 || output_shape32[3] == 3);
    output_h_ = model_out_hwc ? output_shape32[1] : output_shape32[2];
    output_w_ = model_out_hwc ? output_shape32[2] : output_shape32[3];
    output_c_ = model_out_hwc ? output_shape32[3] : output_shape32[1];

    const std::vector<int> output_align_shape =
        tensor_align_shape(download_input_tensor);
    if (output_align_shape.size() != output_shape32.size()) {
        LOG(ERROR) << "download input align_shape rank does not match shape"
            << ", align_shape=" << vec_to_string(output_align_shape)
            << ", shape=" << vec_to_string(to_int_vec(output_shape32));
        return -1;
    }
    align_output_h_ =
        model_out_hwc ? output_align_shape[1] : output_align_shape[2];
    align_output_w_ =
        model_out_hwc ? output_align_shape[2] : output_align_shape[3];
    const size_t align_output_c =
        model_out_hwc ? output_align_shape[3] : output_align_shape[1];
    if (align_output_h_ < output_h_ || align_output_w_ < output_w_ ||
        align_output_c != output_c_) {
        LOG(ERROR) << "invalid download input align_shape"
            << ", align_shape=" << vec_to_string(output_align_shape)
            << ", shape=" << vec_to_string(to_int_vec(output_shape32));
        return -1;
    }

    if (input_h_ < output_h_ || input_w_ < output_w_) {
        LOG(ERROR) << "input size should be greater than output size"
            << ", input_h : " << input_h_ << ", input_w : " << input_w_
            << ", output_h : " << output_h_ << ", output_w : " << output_w_;
        return -1;
    }
    if (image_width <= 0 || image_height <= 0) {
        LOG(ERROR) << "invalid image size " << image_width << "x" << image_height;
        return -1;
    }

    pad_y_ = input_h_ - output_h_;
    pad_x_ = input_w_ - output_w_;
    block_y_size_ = 1 + (static_cast<size_t>(image_height) - 1) / output_h_;
    block_x_size_ = 1 + (static_cast<size_t>(image_width) - 1) / output_w_;

    const xir::Subgraph* root_subgraph = graph_->get_root_subgraph();
    size_t output_canvas_height = 0;
    size_t output_canvas_width = 0;
    if (!read_stride_hw(root_subgraph, "input_stride_shape",
            &input_canvas_height_, &input_canvas_width_) ||
        !read_stride_hw(root_subgraph, "output_stride_shape",
            &output_canvas_height, &output_canvas_width)) {
        return -1;
    }
    if (input_canvas_width_ < static_cast<size_t>(image_width) + pad_x_ ||
        input_canvas_height_ < static_cast<size_t>(image_height) + pad_y_ ||
        output_canvas_width < static_cast<size_t>(image_width) ||
        output_canvas_height < static_cast<size_t>(image_height)) {
        LOG(ERROR) << "image does not fit compiled stride canvas"
            << ", image=" << image_width << "x" << image_height
            << ", input_stride=" << input_canvas_width_ << "x"
            << input_canvas_height_ << ", output_stride=" << output_canvas_width
            << "x" << output_canvas_height;
        return -1;
    }

    output_pixel_bytes_ = output_c_ * output_ele_size_;
    input_row_bytes_ = input_canvas_width_ * input_c_ * input_ele_size_;
    output_row_bytes_ = output_canvas_width * output_pixel_bytes_;
    const size_t input_canvas_size = input_canvas_height_ * input_row_bytes_;
    const size_t output_canvas_size = output_canvas_height * output_row_bytes_;

    LOG(INFO) << "image : " << image_width << "x" << image_height;
    LOG(INFO) << "model input : " << input_w_ << "x" << input_h_
        << " c=" << input_c_;
    LOG(INFO) << "model output : " << output_w_ << "x" << output_h_
        << " c=" << output_c_;
    LOG(INFO) << "model output align : " << align_output_w_ << "x"
        << align_output_h_ << " c=" << align_output_c;
    LOG(INFO) << "block_y_size : " << block_y_size_
        << ", block_x_size : " << block_x_size_;
    LOG(INFO) << "canvas : input " << input_canvas_width_ << " x "
        << input_canvas_height_ << ", output " << output_canvas_width << " x "
        << output_canvas_height;
    LOG(INFO) << "output canvas memory : row_bytes=" << output_row_bytes_;
    LOG(INFO) << "DPU interface : input " << input_reg_name
        << " size=" << input_reg_size_ << " ddr_addr=" << input_ddr_addr_
        << ", output " << output_reg_name << " size=" << output_reg_size_
        << " ddr_addr=" << output_ddr_addr_;

    const size_t num_tiles = block_y_size_ * block_x_size_;
    for (size_t i = 0; i < num_tiles; ++i) {
        std::unique_ptr<XmodelRunner> runner =
            XmodelRunner::create_instance(graph_.get(), options);
        if (runner == nullptr) {
            LOG(ERROR) << "fail to create onnx runner for : " << model;
            return -1;
        }
        runner_list_.emplace_back(std::move(runner));
    }

    const size_t last_block_y = block_y_size_ - 1;
    const size_t last_block_x = block_x_size_ - 1;
    const size_t last_input_bo_end =
        last_block_y * output_h_ * input_row_bytes_ +
        last_block_x * output_w_ * input_c_ * input_ele_size_ + input_reg_size_;
    const size_t last_output_bo_end =
        last_block_y * output_h_ * output_row_bytes_ +
        last_block_x * output_w_ * output_pixel_bytes_ + output_reg_size_;

    constexpr size_t kAlignSize = 64 * 1024;
    input_image_size_ =
        std::max(input_ddr_addr_ + input_canvas_size, last_input_bo_end);
    output_image_size_ =
        std::max(output_ddr_addr_ + output_canvas_size, last_output_bo_end);
    input_parent_buf_ =
        _aligned_malloc(align_up(input_image_size_, kAlignSize), kAlignSize);
    output_parent_buf_ =
        _aligned_malloc(align_up(output_image_size_, kAlignSize), kAlignSize);
    if (input_parent_buf_ == nullptr || output_parent_buf_ == nullptr) {
        LOG(ERROR) << "failed to allocate parent buffers";
        return -1;
    }
    memset(reinterpret_cast<char*>(input_parent_buf_), 0,
        align_up(input_image_size_, kAlignSize));
    memset(reinterpret_cast<char*>(output_parent_buf_), 0,
        align_up(output_image_size_, kAlignSize));
    input_canvas_ =
        reinterpret_cast<unsigned char*>(input_parent_buf_) + input_ddr_addr_;
    output_canvas_ =
        reinterpret_cast<unsigned char*>(output_parent_buf_) + output_ddr_addr_;
    return 0;
}

bool XmodelZeroCopySession::create_bos() {
    std::vector<void*> input_ptrs(1, input_parent_buf_);
    std::vector<size_t> input_sizes(1, input_image_size_);
    parent_input_bo_ =
        runner_list_[0]->create_parent_bo_input(input_ptrs, input_sizes);
    std::vector<void*> output_ptrs(1, output_parent_buf_);
    std::vector<size_t> output_sizes(1, output_image_size_);
    parent_output_bo_ =
        runner_list_[0]->create_parent_bo_output(output_ptrs, output_sizes);
    if (parent_input_bo_ == nullptr || parent_output_bo_ == nullptr) {
        LOG(ERROR) << "failed to create parent input/output BOs";
        return false;
    }

    const size_t num_tiles = runner_list_.size();
    for (size_t i = 0; i < num_tiles; ++i) {
        const size_t block_y = i / block_x_size_;
        const size_t block_x = i % block_x_size_;

        const size_t input_bo_offset = block_y * output_h_ * input_row_bytes_ +
            block_x * output_w_ * input_c_ * input_ele_size_;
        xrt::bo* sub_input_bo = runner_list_[i]->create_sub_bo_input(
            parent_input_bo_, input_reg_size_, input_bo_offset);
        if (sub_input_bo == nullptr) {
            LOG(ERROR) << "failed to create input sub-BO " << i;
            return false;
        }
        sub_input_bos_.push_back(sub_input_bo);

        const size_t output_bo_offset =
            block_y * output_h_ * output_row_bytes_ +
            block_x * output_w_ * output_pixel_bytes_;
        xrt::bo* sub_output_bo = runner_list_[i]->create_sub_bo_output(
            parent_output_bo_, output_reg_size_, output_bo_offset);
        if (sub_output_bo == nullptr) {
            LOG(ERROR) << "failed to create output sub-BO " << i;
            return false;
        }
        sub_output_bos_.push_back(sub_output_bo);
    }
    return true;
}

void XmodelZeroCopySession::input_sync() {
    runner_list_[0]->input_sync(parent_input_bo_, parent_output_bo_);
}

void XmodelZeroCopySession::add_to_runlist() {
    for (size_t i = 0; i < runner_list_.size(); ++i) {
        runner_list_[i]->add_to_runlist(0);
    }
}

void XmodelZeroCopySession::execute_runlist() {
    runner_list_[runner_list_.size() - 1]->execute_runlist(0);
}

void XmodelZeroCopySession::output_sync() {
    runner_list_[runner_list_.size() - 1]->output_sync(
        sub_output_bos_[sub_output_bos_.size() - 1]);
}

} // namespace tiling_sr
