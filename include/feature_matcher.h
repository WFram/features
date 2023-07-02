//
// Created by wfram on 2/22/23.
//

#ifndef WF_FEATURE_EXTRACTOR_FEATURE_MATCHER_H
#define WF_FEATURE_EXTRACTOR_FEATURE_MATCHER_H

#include <map>

#include <opencv4/opencv2/core.hpp>

#include <utils.h>

namespace feature_matcher {

using FeatureCorrespondences = std::map<size_t, size_t>;

class FeatureMatcher {
 public:
  FeatureMatcher() = default;

  virtual void match(const std::vector<cv::KeyPoint> &keypoints1, const cv::Mat &descriptors1,
                     const std::vector<cv::KeyPoint> &keypoints2, const cv::Mat &descriptors2,
                     FeatureCorrespondences &correspondences) = 0;
};

class BFFeaturematcher : public FeatureMatcher {
 public:
  using ImagePyramid = std::vector<cv::Mat>;

  BFFeaturematcher(const ImagePyramid &image_pyramid_from, const ImagePyramid &image_pyramid_to,
                   const std::vector<Precision> &scale_factor_per_level, const int &high_threshold,
                   const int &low_threshold, const Precision &lowe_ratio)
      : image_pyramid_from_(image_pyramid_from),
        image_pyramid_to_(image_pyramid_to),
        upscale_vector_(scale_factor_per_level),
        high_threshold_(high_threshold),
        low_threshold_(low_threshold),
        lowe_ratio_(lowe_ratio) {}

  int descriptorDistance(const cv::Mat &descriptor1, const cv::Mat &descriptor2);

  void match(const std::vector<cv::KeyPoint> &keypoints1, const cv::Mat &descriptors1,
             const std::vector<cv::KeyPoint> &keypoints2, const cv::Mat &descriptors2,
             FeatureCorrespondences &correspondences) override;

 protected:
  const ImagePyramid &image_pyramid_from_;
  const ImagePyramid &image_pyramid_to_;
  const std::vector<Precision> &upscale_vector_;
  const int high_threshold_;
  const int low_threshold_;
  const Precision lowe_ratio_;
};

}  // namespace feature_matcher

#endif  // WF_FEATURE_EXTRACTOR_FEATURE_MATCHER_H
