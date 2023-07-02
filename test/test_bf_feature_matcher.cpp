//
// Created by wfram on 2/26/23.
//

#include <opencv4/opencv2/core.hpp>

#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>
#include <fstream>

#include "feature_extractor.h"
#include "feature_matcher.h"

namespace feature_matcher {

class TestBFFeatureMatcher : public ::testing::Test {
 public:
  void SetUp() override {
    test_file_from_ = TEST_DATA_DIR "00000.jpg";
    test_file_to_ = TEST_DATA_DIR "00001.jpg";
    // TODO: make configs here
  }

 protected:
  std::string test_file_from_;
  std::string test_file_to_;
};

// TODO: test detector and extractor separately
TEST_F(TestBFFeatureMatcher, testBFFeatureMatcher) {
  int number_of_features = 1500;
  size_t number_of_pyramid_levels = 8;
  Precision scale_factor = 1.2_p;
  int edge_threshold = 19;

  int high_threshold = 100;
  int low_threshold = 50;
  Precision lowe_ratio = 1.0_p;

  cv::Mat image_from = cv::imread(test_file_from_, cv::IMREAD_GRAYSCALE);
  cv::Mat image_to = cv::imread(test_file_to_, cv::IMREAD_GRAYSCALE);
  cv::Mat debug_image;

  const auto& upscale_vector = utils::computeUpScaleVector(number_of_pyramid_levels, scale_factor);
  const auto& image_pyramid_from = utils::computeImagePyramid(image_from, upscale_vector, edge_threshold);
  const auto& image_pyramid_to = utils::computeImagePyramid(image_to, upscale_vector, edge_threshold);

  orb_feature_extractor::ORBFeatureExtractor feature_extractor(number_of_features, upscale_vector);
  orb_feature_extractor::Keypoints keypoints_from, keypoints_to;
  cv::Mat descriptors_from, descriptors_to;
  feature_extractor.extract(image_pyramid_from, keypoints_from, descriptors_from);
  feature_extractor.extract(image_pyramid_to, keypoints_to, descriptors_to);

  feature_matcher::BFFeaturematcher feature_matcher(image_pyramid_from, image_pyramid_to, upscale_vector,
                                                    high_threshold, low_threshold, lowe_ratio);
  FeatureCorrespondences correspondences;
  feature_matcher.match(keypoints_from, descriptors_from, keypoints_to, descriptors_to, correspondences);
  EXPECT_NE(correspondences.size(), 0);

  std::vector<cv::DMatch> debug_matches;
  std::ofstream log_stream("log.txt", std::ios::trunc);
  for (const auto& correspondence : correspondences) {
    log_stream << correspondence.first << " " << correspondence.second << "\n";
  }
  for (const auto& match : correspondences) {
    debug_matches.emplace_back(match.first, match.second, 1.0);
  }
  // TODO: bad matches, debug please

  // TODO: find the reason why so little matches
  cv::namedWindow("test", cv::WINDOW_NORMAL);
  cv::resizeWindow("test", 1980, 1280);
  cv::drawMatches(image_pyramid_from[0], keypoints_from, image_pyramid_to[0], keypoints_to, debug_matches, debug_image,
                  cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 0));
  cv::imshow("test", debug_image);
  cv::waitKey(0);
}

}  // namespace feature_matcher