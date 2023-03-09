//
// Created by wfram on 2/23/23.
//

#include <utils.h>

// TODO: do we need the pointer or use const ref??
std::vector<Precision> utils::computeUpScaleVector(const size_t &number_of_pyramid_levels,
                                                   const Precision &scale_factor) {
  std::vector<Precision> upscale_factor_per_level(number_of_pyramid_levels);
  upscale_factor_per_level[0] = 1.0_p;

  for (size_t level = 1; level < number_of_pyramid_levels; level++)
    upscale_factor_per_level[level] = upscale_factor_per_level[level - 1] * scale_factor;

  return upscale_factor_per_level;
}

std::unique_ptr<ImagePyramid> utils::computeImagePyramid(const cv::Mat &image,
                                                         const std::vector<Precision> &upscale_vector,
                                                         const int &edge_threshold) {
  ImagePyramid image_pyramid(upscale_vector.size());
  std::vector<Precision> downscale_factor_per_level(upscale_vector.size());
  downscale_factor_per_level[0] = 1.0_p;

  // TODO: write scale vectors creation
  for (size_t level = 1; level < upscale_vector.size(); level++)
    downscale_factor_per_level[level] = 1.0_p / upscale_vector[level];

  for (size_t level = 0; level < upscale_vector.size(); ++level) {
    Precision scale = downscale_factor_per_level[level];
    // TODO: rename
    cv::Size size(cvRound(static_cast<Precision>(image.cols) * scale),
                  cvRound(static_cast<Precision>(image.rows) * scale));
    cv::Size whole_size(size.width + edge_threshold * 2, size.height + edge_threshold * 2);
    cv::Mat temporal_image(whole_size, image.type());
    image_pyramid[level] = temporal_image(cv::Rect(edge_threshold, edge_threshold, size.width, size.height));

    // Compute the resized image
    if (level != 0) {
      cv::resize(image_pyramid[level - 1], image_pyramid[level], size, 0, 0, cv::INTER_LINEAR);

      cv::copyMakeBorder(image_pyramid[level], temporal_image, edge_threshold, edge_threshold, edge_threshold,
                         edge_threshold, cv::BORDER_REFLECT_101 + cv::BORDER_ISOLATED);
    } else {
      cv::copyMakeBorder(image, temporal_image, edge_threshold, edge_threshold, edge_threshold, edge_threshold,
                         cv::BORDER_REFLECT_101);
    }
  }

  return std::make_unique<ImagePyramid>(image_pyramid);
}
