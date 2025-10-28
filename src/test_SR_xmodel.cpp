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
#include <cmath>
#include <cstring>
#include <string>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "opencv2/opencv.hpp"

#include "xmodel_zero_copy.h"

DEFINE_string(model, "", "onnx model path");
DEFINE_string(image, "", "image file");
DEFINE_string(vai_options, "", "");
DEFINE_string(output, "sr.png", "save output image");
DEFINE_int32(input_zero_point, 0, "pre-process zero_point");
DEFINE_double(input_y_scale, 1.0,
    "pre-process y_scale: q = (uint8 - zero_point) / y_scale");
DEFINE_int32(output_zero_point, 84, "post-process zero_point");
DEFINE_double(output_y_scale, 4.0,
    "post-process y_scale: uint8 = (q - zero_point) * y_scale");

static uint8_t preprocess_pixel(uint8_t pixel) {
    const float value =
        (static_cast<float>(pixel) - static_cast<float>(FLAGS_input_zero_point)) /
        static_cast<float>(FLAGS_input_y_scale);
    return static_cast<uint8_t>(std::clamp(std::lround(value), 0L, 255L));
}

static uint8_t postprocess_pixel(uint8_t raw_value) {
    const float value = (static_cast<float>(raw_value) -
        static_cast<float>(FLAGS_output_zero_point)) *
        static_cast<float>(FLAGS_output_y_scale);
    return static_cast<uint8_t>(std::clamp(std::lround(value), 0L, 255L));
}

static void write_pixel(unsigned char* dst, size_t input_c,
    const cv::Mat& image, int y, int x) {
    if (image.channels() == 3) {
        cv::Vec3b pixels = image.at<cv::Vec3b>(y, x);
        dst[0] = preprocess_pixel(pixels[0]);
        dst[1] = preprocess_pixel(pixels[1]);
        dst[2] = preprocess_pixel(pixels[2]);
    }
    else {
        cv::Vec4b pixels = image.at<cv::Vec4b>(y, x);
        dst[0] = preprocess_pixel(pixels[0]);
        dst[1] = preprocess_pixel(pixels[1]);
        dst[2] = preprocess_pixel(pixels[2]);
    }
    dst[3] = 0;
}

static void fill_input_canvas(tiling_sr::XmodelZeroCopySession* session,
    const cv::Mat& image) {
    const size_t row_bytes =
        session->input_canvas_width() * session->input_c() * session->input_ele_size();
    auto* base = session->input_canvas();
    std::memset(base, preprocess_pixel(0),
        session->input_canvas_height() * row_bytes);

    const size_t image_offset_y = session->pad_y() / 2;
    const size_t image_offset_x = session->pad_x() / 2;
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            unsigned char* dst =
                base + (static_cast<size_t>(y) + image_offset_y) * row_bytes +
                (static_cast<size_t>(x) + image_offset_x) * session->input_c() *
                session->input_ele_size();
            write_pixel(dst, session->input_c(), image, y, x);
        }
    }
}

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    if (FLAGS_model == "" || FLAGS_image == "") {
        LOG(ERROR) << "invalid model or patch";
        return -1;
    }

    cv::Mat image = cv::imread(FLAGS_image, cv::IMREAD_UNCHANGED);
    if (image.empty()) {
        LOG(ERROR) << "fail to read image : " << FLAGS_image;
        return -1;
    }

    auto session = tiling_sr::XmodelZeroCopySession::open(
        FLAGS_model, tiling_sr::parse_vai_options(FLAGS_vai_options),
        image.cols, image.rows);
    if (session == nullptr) {
        return -1;
    }
    if ((image.channels() != 3 && image.channels() != 4) ||
        (session->input_c() != 3 && session->input_c() != 4)) {
        LOG(ERROR) << "unmatched shape" << ", image channel : " << image.channels()
            << ", model channel : " << session->input_c();
        return -1;
    }
    if (session->output_c() != 3 && session->output_c() != 4) {
        LOG(ERROR) << "invalid output channel : " << session->output_c();
        return -1;
    }

    LOG(INFO) << "image : " << image.cols << "x" << image.rows
        << " ch=" << image.channels();
    LOG(INFO) << "pre-process  : zero_point=" << FLAGS_input_zero_point
        << ", y_scale=" << FLAGS_input_y_scale;
    LOG(INFO) << "post-process : zero_point=" << FLAGS_output_zero_point
        << ", y_scale=" << FLAGS_output_y_scale;

    fill_input_canvas(session.get(), image);
    if (!session->create_bos()) {
        return -1;
    }

    session->input_sync();
    session->add_to_runlist();
    session->execute_runlist();
    session->output_sync();

    cv::Mat cropped_image(image.rows, image.cols, CV_8UC4);
    const auto* output_base = session->output_canvas();
    for (int y = 0; y < cropped_image.rows; ++y) {
        for (int x = 0; x < cropped_image.cols; ++x) {
            const unsigned char* src = output_base +
                static_cast<size_t>(y) * session->output_row_bytes() +
                static_cast<size_t>(x) * session->output_pixel_bytes();
            cv::Vec4b pixel;
            pixel[0] = postprocess_pixel(src[0]);
            pixel[1] = postprocess_pixel(src[1]);
            pixel[2] = postprocess_pixel(src[2]);
            pixel[3] = 255;
            cropped_image.at<cv::Vec4b>(y, x) = pixel;
        }
    }

    cv::imwrite(FLAGS_output, cropped_image);
    LOG(INFO) << "save image to : " << FLAGS_output;
    return 0;
}
