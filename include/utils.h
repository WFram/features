//
// Created by wfram on 2/23/23.
//

#ifndef WF_FEATURE_EXTRACTOR_UTILS_H
#define WF_FEATURE_EXTRACTOR_UTILS_H

#include <opencv4/opencv2/opencv.hpp>

using Precision = float;

constexpr Precision operator"" _p(long double arg) { return static_cast<Precision>(arg); }

using ImagePyramid = std::vector<cv::Mat>;

namespace utils {

// TODO: add
std::vector<Precision> computeUpScaleVector(const size_t &number_of_pyramid_levels, const Precision &scale_factor);

std::unique_ptr<ImagePyramid> computeImagePyramid(const cv::Mat &image, const std::vector<Precision> &upscale_vector,
                                                  const int &edge_threshold);

}  // namespace utils

#endif  // WF_FEATURE_EXTRACTOR_UTILS_H
