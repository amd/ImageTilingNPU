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

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "opencv2/opencv.hpp"

#include "BS_thread_pool.hpp"
#include "onnx_runner.h"

DEFINE_string(model, "", "onnx model path");
DEFINE_int32(threads, 4, "num of perf threads");
DEFINE_string(vai_options, "", "");
DEFINE_string(image, "", "image file");
DEFINE_string(output, "sr_onnx.png", "output image path");
DEFINE_bool(vitisai, false, "use VitisAI execution provider; default CPU");
DEFINE_int32(input_zero_point, 0, "pre-process zero_point");
DEFINE_double(input_y_scale, 256.0, "pre-process y_scale: float = (uint8 - zero_point) / y_scale");
DEFINE_int32(output_zero_point, 0, "post-process zero_point");
DEFINE_double(output_y_scale, 256.0, "post-process y_scale: uint8 = (float - zero_point) * y_scale");

static float preprocess_to_float(uint8_t pixel) {
    return (static_cast<float>(pixel) - static_cast<float>(FLAGS_input_zero_point)) /
        static_cast<float>(FLAGS_input_y_scale);
}

static uint8_t preprocess_pixel(uint8_t pixel) {
    const float value = preprocess_to_float(pixel);
    return static_cast<uint8_t>(std::clamp(std::lround(value), 0L, 255L));
}

static uint8_t postprocess_int8_pixel(uint8_t raw_value) {
    const float value = (static_cast<float>(raw_value) - static_cast<float>(FLAGS_output_zero_point)) *
        static_cast<float>(FLAGS_output_y_scale);
    return static_cast<uint8_t>(std::clamp(std::lround(value), 0L, 255L));
}

static uint8_t postprocess_float_pixel(float value) {
    const float normalized = (value - static_cast<float>(FLAGS_output_zero_point)) *
        static_cast<float>(FLAGS_output_y_scale);
    return static_cast<uint8_t>(std::clamp(std::lround(normalized), 0L, 255L));
}

template<typename T>
struct TypeTag {
    using type = T;
};

template<typename InputDataType, typename Callback>
void dispatch_output_type(tiling_sr::TensorInfo::DataType output_data_type, Callback&& callback) {
    switch (output_data_type) {
    case tiling_sr::TensorInfo::DataType::FLOAT32:
        callback(TypeTag<InputDataType>{}, TypeTag<float>{});
        break;
    case tiling_sr::TensorInfo::DataType::UINT8:
        callback(TypeTag<InputDataType>{}, TypeTag<uint8_t>{});
        break;
    case tiling_sr::TensorInfo::DataType::INT8:
        callback(TypeTag<InputDataType>{}, TypeTag<int8_t>{});
        break;
    }
}

template<typename Callback>
void dispatch_tiling_types(
    tiling_sr::TensorInfo::DataType input_data_type,
    tiling_sr::TensorInfo::DataType output_data_type,
    Callback&& callback) {
    switch (input_data_type) {
    case tiling_sr::TensorInfo::DataType::FLOAT32:
        dispatch_output_type<float>(output_data_type, std::forward<Callback>(callback));
        break;
    case tiling_sr::TensorInfo::DataType::UINT8:
        dispatch_output_type<uint8_t>(output_data_type, std::forward<Callback>(callback));
        break;
    case tiling_sr::TensorInfo::DataType::INT8:
        dispatch_output_type<int8_t>(output_data_type, std::forward<Callback>(callback));
        break;
    }
}

template<typename InputDataType, typename OutputDataType>
void tiling_process(char* input_image, char* output_image, char* input_buf, char* output_buf,
    size_t block_x_size, size_t block_y_size,
    size_t input_canvas_width, size_t input_h, size_t input_w, size_t input_c, size_t input_ele_size,
    size_t output_canvas_width, size_t output_h, size_t output_w, size_t output_c, size_t output_ele_size,
    std::atomic<int>& global_task_id, tiling_sr::BaseRunner* runner) {

    while (1) {
        int current_task_id = global_task_id.fetch_add(1);
        if (current_task_id >= block_x_size * block_y_size) {
            break;
        }
        size_t block_x = current_task_id % block_x_size;
        size_t block_y = current_task_id / block_x_size;

        char* onnx_input_addr = input_buf + input_w * input_h * input_c * input_ele_size * current_task_id;
        const size_t input_x = block_x * output_w;
        const size_t input_y = block_y * output_h;
        for (size_t y = 0; y < input_h; ++y) {
            char* input_block_addr = input_image + ((input_y + y) * input_canvas_width + input_x) * input_c;
            const size_t copy_len = input_w * input_c;
            if (input_ele_size == 1) {
                unsigned char* input_dst_addr = reinterpret_cast<unsigned char*>(onnx_input_addr) + y * input_w * input_c;
                for (size_t j = 0; j < copy_len; ++j) {
                    input_dst_addr[j] = preprocess_pixel(static_cast<unsigned char>(input_block_addr[j]));
                }
            }
            else {
                InputDataType* temp_dst = reinterpret_cast<InputDataType*>(onnx_input_addr) + y * input_w * input_c;
                for (size_t j = 0; j < copy_len; ++j) {
                    temp_dst[j] = static_cast<InputDataType>(
                        preprocess_to_float(static_cast<unsigned char>(input_block_addr[j])));
                }
            }
        }
        std::vector<tiling_sr::OrtBuf> input(1);
        input[0] = { onnx_input_addr, input_h * input_w * input_c * input_ele_size };
        char* onnx_output_addr = output_buf + output_w * output_h * output_c * output_ele_size * current_task_id;
        std::vector<tiling_sr::OrtBuf> output(1);
        output[0] = { onnx_output_addr, output_h * output_w * output_c * output_ele_size };
        if (runner->run(input, output) != 0) {
            LOG(ERROR) << "onnx run failed for tile " << current_task_id;
            continue;
        }
        for (size_t i = 0; i < output_h; ++i) {
            char* output_block_addr = output_image + ((i + block_y * output_h) * output_canvas_width + block_x * output_w) * output_c;
            size_t copy_len = output_w * output_c;
            if (output_ele_size == 1) {
                const unsigned char* src = reinterpret_cast<const unsigned char*>(onnx_output_addr + i * output_w * output_c * output_ele_size);
                for (size_t j = 0; j < copy_len; ++j) {
                    output_block_addr[j] = static_cast<char>(postprocess_int8_pixel(src[j]));
                }
            }
            else {
                OutputDataType* temp_src = reinterpret_cast<OutputDataType*>(onnx_output_addr + i * output_w * output_c * output_ele_size);
                for (size_t j = 0; j < copy_len; ++j) {
                    output_block_addr[j] = static_cast<char>(postprocess_float_pixel(static_cast<float>(temp_src[j])));
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    if (FLAGS_model == "" || FLAGS_image == "") {
        LOG(ERROR) << "invalid model or image";
        return -1;
    }

    const bool use_vitisai = FLAGS_vitisai;
    LOG(INFO) << "execution provider : " << (use_vitisai ? "vitisai" : "cpu");

    std::unordered_map<std::string, std::string> options;
    do {
        std::istringstream iss(FLAGS_vai_options);
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
    } while (0);

    std::vector<int64_t> input_shape;
    std::vector<int64_t> output_shape;
    size_t input_ele_size = 0;
    size_t output_ele_size = 0;
    size_t input_tensor_size = 0;
    size_t output_tensor_size = 0;
    tiling_sr::TensorInfo::DataType input_data_type;
    tiling_sr::TensorInfo::DataType output_data_type;
    do {
        std::unique_ptr<tiling_sr::BaseRunner> temp_runner =
            tiling_sr::ONNXRunner::create_instance(FLAGS_model, false, options);
        if (temp_runner == nullptr) {
            LOG(ERROR) << "fail to create onnx runner for : " << FLAGS_model;
            return -1;
        }

        std::vector<tiling_sr::TensorInfo> in_s;
        std::vector<tiling_sr::TensorInfo> out_s;
        temp_runner->get_runner_io(in_s, out_s);
        if (in_s.size() != 1) {
            LOG(ERROR) << "model should have only one input";
            return -1;
        }
        input_shape = in_s[0].shape;
        input_data_type = in_s[0].data_type;
        input_ele_size = in_s[0].ele_size();
        input_tensor_size = in_s[0].tensor_size();
        if (out_s.size() != 1) {
            LOG(ERROR) << "model should have only one output";
            return -1;
        }
        output_shape = out_s[0].shape;
        output_data_type = out_s[0].data_type;
        output_ele_size = out_s[0].ele_size();
        output_tensor_size = out_s[0].tensor_size();
    } while (0);

    bool model_in_hwc = (input_shape[3] == 4 || input_shape[3] == 3);
    int64_t input_h = model_in_hwc ? input_shape[1] : input_shape[2];
    int64_t input_w = model_in_hwc ? input_shape[2] : input_shape[3];
    int64_t input_c = model_in_hwc ? input_shape[3] : input_shape[1];
    bool model_out_hwc = (output_shape[3] == 4 || output_shape[3] == 3);
    int64_t output_h = model_out_hwc ? output_shape[1] : output_shape[2];
    int64_t output_w = model_out_hwc ? output_shape[2] : output_shape[3];
    int64_t output_c = model_out_hwc ? output_shape[3] : output_shape[1];

    cv::Mat image = cv::imread(FLAGS_image, cv::IMREAD_UNCHANGED);
    if (image.empty()) {
        LOG(ERROR) << "fail to read image : " << FLAGS_image;
        return -1;
    }
    if (image.channels() != 3 && image.channels() != 4) {
        LOG(ERROR) << "invalid image channel : " << image.channels();
        return -1;
    }

    if (input_h < output_h || input_w < output_w) {
        LOG(ERROR) << "input_h : " << input_h
            << ", input_w : " << input_w
            << ", output_h : " << output_h
            << ", output_w : " << output_w;
        return -1;
    }

    std::vector<std::unique_ptr<tiling_sr::BaseRunner>> runner_list;
    for (size_t i = 0; i < static_cast<size_t>(FLAGS_threads); ++i) {
        std::unique_ptr<tiling_sr::BaseRunner> runner =
            tiling_sr::ONNXRunner::create_instance(FLAGS_model, use_vitisai, options);
        if (runner == nullptr) {
            LOG(ERROR) << "fail to create onnx runner for : " << FLAGS_model;
            return -1;
        }
        runner_list.emplace_back(std::move(runner));
    }

    size_t output_image_width = image.cols;
    size_t output_image_height = image.rows;
    size_t block_y_size = 1 + (output_image_height - 1) / output_h;
    size_t block_x_size = 1 + (output_image_width - 1) / output_w;
    size_t input_padding_y = input_h - output_h;
    size_t input_padding_x = input_w - output_w;
    size_t input_canvas_width = block_x_size * output_w + input_padding_x;
    size_t input_canvas_height = block_y_size * output_h + input_padding_y;
    size_t output_canvas_width = block_x_size * output_w;
    size_t output_canvas_height = block_y_size * output_h;
    LOG(INFO) << "block_y_size : " << block_y_size
        << ", block_x_size : " << block_x_size;
    LOG(INFO) << "pre-process  : zero_point=" << FLAGS_input_zero_point
        << ", y_scale=" << FLAGS_input_y_scale;
    LOG(INFO) << "post-process : zero_point=" << FLAGS_output_zero_point
        << ", y_scale=" << FLAGS_output_y_scale;

    std::unique_ptr<char[]> input_buf(new char[input_canvas_width * input_canvas_height * input_c]);
    memset(reinterpret_cast<char*>(input_buf.get()), 0, input_canvas_width * input_canvas_height * input_c);
    for (size_t i = 0; i < image.rows; ++i) {
        for (size_t j = 0; j < image.cols; ++j) {
            size_t offset = (i + input_padding_y / 2) * input_canvas_width + j + input_padding_x / 2;
            unsigned char* block_buf = reinterpret_cast<unsigned char*>(input_buf.get()) + offset * input_c;
            if (image.channels() == 3) {
                cv::Vec3b pixels = image.at<cv::Vec3b>(i, j);
                block_buf[0] = pixels[0];
                block_buf[1] = pixels[1];
                block_buf[2] = pixels[2];
                if (input_c == 4) {
                    block_buf[3] = 0;
                }
            }
            else {
                cv::Vec4b pixels = image.at<cv::Vec4b>(i, j);
                block_buf[0] = pixels[0];
                block_buf[1] = pixels[1];
                block_buf[2] = pixels[2];
                if (input_c == 4) {
                    block_buf[3] = 0;
                }
            }
        }
    }
    std::unique_ptr<char[]> output_buf(new char[output_canvas_width * output_canvas_height * output_c]);
    memset(reinterpret_cast<char*>(output_buf.get()), 0, output_canvas_width * output_canvas_height * output_c);

    std::unique_ptr<char[]> runner_input_buf(new char[block_y_size * block_x_size * input_tensor_size]);
    memset(runner_input_buf.get(), 0, block_y_size * block_x_size * input_tensor_size);
    std::unique_ptr<char[]> runner_output_buf(new char[block_y_size * block_x_size * output_tensor_size]);
    memset(runner_output_buf.get(), 0, block_y_size * block_x_size * output_tensor_size);

    BS::thread_pool pool(FLAGS_threads);
    BS::multi_future<void> mf(FLAGS_threads);
    std::atomic<int> task_index(0);
    for (size_t i = 0; i < static_cast<size_t>(FLAGS_threads); ++i) {
        dispatch_tiling_types(input_data_type, output_data_type, [&](auto input_type_tag, auto output_type_tag) {
            using InputDataType = typename decltype(input_type_tag)::type;
            using OutputDataType = typename decltype(output_type_tag)::type;
            mf[i] = pool.submit(tiling_process<InputDataType, OutputDataType>, input_buf.get(), output_buf.get(), runner_input_buf.get(), runner_output_buf.get(),
                block_x_size, block_y_size,
                input_canvas_width, input_h, input_w, input_c, input_ele_size,
                output_canvas_width, output_h, output_w, output_c, output_ele_size,
                std::ref(task_index), runner_list[i].get());
            });
    }
    mf.wait();

    cv::Mat output_canvas(output_canvas_height, output_canvas_width, CV_8UC4, output_buf.get());
    cv::Mat output_image = output_canvas(cv::Rect(0, 0, output_image_width, output_image_height));
    for (int y = 0; y < output_image.rows; ++y) {
        for (int x = 0; x < output_image.cols; ++x) {
            output_image.at<cv::Vec4b>(y, x)[3] = 255;
        }
    }
    cv::imwrite(FLAGS_output, output_image);

    LOG(INFO) << "save image to : " << FLAGS_output;

    return 0;
}
