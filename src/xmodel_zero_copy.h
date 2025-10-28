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

#ifndef _XMODEL_ZERO_COPY_H_
#define _XMODEL_ZERO_COPY_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "xrt/xrt_bo.h"

#include "xmodel_runner.h"

namespace tiling_sr {

std::unordered_map<std::string, std::string> parse_vai_options(
    const std::string& vai_options);

class XmodelZeroCopySession {
public:
    static std::unique_ptr<XmodelZeroCopySession> open(
        const std::string& model,
        const std::unordered_map<std::string, std::string>& options,
        int image_width, int image_height);
    ~XmodelZeroCopySession();

    XmodelZeroCopySession(const XmodelZeroCopySession&) = delete;
    XmodelZeroCopySession& operator=(const XmodelZeroCopySession&) = delete;

    bool create_bos();
    void input_sync();
    void add_to_runlist();
    void execute_runlist();
    void output_sync();

    unsigned char* input_canvas() const { return input_canvas_; }
    unsigned char* output_canvas() const { return output_canvas_; }

    size_t input_h() const { return input_h_; }
    size_t input_w() const { return input_w_; }
    size_t input_c() const { return input_c_; }
    size_t input_ele_size() const { return input_ele_size_; }
    size_t output_h() const { return output_h_; }
    size_t output_w() const { return output_w_; }
    size_t output_c() const { return output_c_; }
    size_t output_ele_size() const { return output_ele_size_; }
    size_t align_output_h() const { return align_output_h_; }
    size_t align_output_w() const { return align_output_w_; }
    size_t pad_y() const { return pad_y_; }
    size_t pad_x() const { return pad_x_; }
    size_t input_canvas_width() const { return input_canvas_width_; }
    size_t input_canvas_height() const { return input_canvas_height_; }
    size_t output_row_bytes() const { return output_row_bytes_; }
    size_t output_pixel_bytes() const { return output_pixel_bytes_; }

private:
    XmodelZeroCopySession() = default;
    int init(const std::string& model,
        const std::unordered_map<std::string, std::string>& options,
        int image_width, int image_height);

    std::unique_ptr<xir::Graph> graph_;
    std::vector<std::unique_ptr<XmodelRunner> > runner_list_;
    void* input_parent_buf_ = nullptr;
    void* output_parent_buf_ = nullptr;
    unsigned char* input_canvas_ = nullptr;
    unsigned char* output_canvas_ = nullptr;
    xrt::bo* parent_input_bo_ = nullptr;
    xrt::bo* parent_output_bo_ = nullptr;
    std::vector<xrt::bo*> sub_input_bos_;
    std::vector<xrt::bo*> sub_output_bos_;

    size_t input_h_ = 0;
    size_t input_w_ = 0;
    size_t input_c_ = 0;
    size_t input_ele_size_ = 0;
    size_t output_h_ = 0;
    size_t output_w_ = 0;
    size_t output_c_ = 0;
    size_t output_ele_size_ = 0;
    size_t align_output_h_ = 0;
    size_t align_output_w_ = 0;
    size_t pad_y_ = 0;
    size_t pad_x_ = 0;
    size_t block_y_size_ = 0;
    size_t block_x_size_ = 0;
    size_t input_canvas_width_ = 0;
    size_t input_canvas_height_ = 0;
    size_t input_row_bytes_ = 0;
    size_t output_row_bytes_ = 0;
    size_t output_pixel_bytes_ = 0;
    size_t input_ddr_addr_ = 0;
    size_t output_ddr_addr_ = 0;
    size_t input_reg_size_ = 0;
    size_t output_reg_size_ = 0;
    size_t input_image_size_ = 0;
    size_t output_image_size_ = 0;
};

} // namespace tiling_sr

#endif // _XMODEL_ZERO_COPY_H_
