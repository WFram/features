#include <iostream>
#include <vector>

#include <opencv4/opencv2/core.hpp>

#include <gtest/gtest.h>

#include "feature_extractor.h"

namespace orb_feature_extractor {

class TestORBFeatureExtractor : public ::testing::Test {
 public:
  void SetUp() override { test_file_ = TEST_DATA_DIR "00000.jpg"; }

 protected:
  std::string test_file_;
};

TEST_F(TestORBFeatureExtractor, testORBFeatureExtractor) {
  cv::Mat image = cv::imread(test_file_, cv::IMREAD_GRAYSCALE);
  cv::Mat debug_image;

  // TODO: set from YAML file
  orb_feature_extractor::ORBFeatureExtractor feature_extractor(1500, 8, 1.2);
  orb_feature_extractor::Keypoints keypoints;
  feature_extractor.extract(image, keypoints);

  cv::drawKeypoints(image, keypoints, debug_image, cv::Scalar(0, 255, 0));
  cv::imshow("debug_image", debug_image);
  cv::waitKey(0);
}

}  // namespace orb_feature_extractor

