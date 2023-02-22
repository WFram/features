//
// Created by wfram on 2/22/23.
//

#ifndef WF_FEATURE_EXTRACTOR_FEATURE_MATCHER_H
#define WF_FEATURE_EXTRACTOR_FEATURE_MATCHER_H

#include <map>

#include <opencv4/opencv2/core.hpp>

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
  int descriptorDistance(const cv::Mat &descriptor1, const cv::Mat &descriptor2);

  void match(const std::vector<cv::KeyPoint> &keypoints1, const cv::Mat &descriptors1,
             const std::vector<cv::KeyPoint> &keypoints2, const cv::Mat &descriptors2,
             FeatureCorrespondences &correspondences) override;
};

}  // namespace feature_matcher

#endif  // WF_FEATURE_EXTRACTOR_FEATURE_MATCHER_H
