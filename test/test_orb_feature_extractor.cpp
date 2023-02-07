#include <iostream>
#include <vector>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>

#include <gtest/gtest.h>

#include "feature_extractor.h"

/* TODO: create an abstract class which performs feature extraction
          it should provide at least two inherited feature extractors: SIFT and ORB
          ORB extractor should be written from the scratch
          for SIFT the standard OpenCV implementation can be used (for now)
          when an instance is created it should pass a type of the feature extractor
          to extract the features it also should have an ability to specify if we need grayscale or color image

 TODO: make as a gtest
       1) set a test dir and take an image from there
       2) take parameters for the extractor by using YAM*/

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

