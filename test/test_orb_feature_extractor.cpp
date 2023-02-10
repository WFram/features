#include <any>
#include <iostream>
#include <vector>

#include <opencv4/opencv2/core.hpp>

#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include "feature_extractor.h"

namespace orb_feature_extractor {

class TestORBFeatureExtractor : public ::testing::Test {
 public:
  void SetUp() override {
    test_file_ = TEST_DATA_DIR "00000.jpg";
    config_file_ = TEST_DATA_DIR "extractor_config.yaml";
    config_node_ = YAML::LoadFile(config_file_);
  }

 protected:
  std::string test_file_;
  std::string config_file_;
  YAML::Node config_node_;
};

TEST_F(TestORBFeatureExtractor, testORBFeatureExtractor) {
  auto feature_extractor_config = config_node_["feature_extractor"];
  std::map<std::string, std::any> parameters;
  for (auto &[key, value] : feature_extractor_config.as<std::map<std::string, YAML::Node>>()) {
    parameters[key] = value.as<int>();
  }

  const auto number_of_features = std::any_cast<int>(parameters.at("number_of_features"));
  const auto number_of_pyramid_levels = std::any_cast<int>(parameters.at("number_of_pyramid_levels"));
  const auto scale_factor = std::any_cast<int>(parameters.at("scale_factor"));

  cv::Mat image = cv::imread(test_file_, cv::IMREAD_GRAYSCALE);
  cv::Mat debug_image;

  // TODO: solve casting issues
  orb_feature_extractor::ORBFeatureExtractor feature_extractor(number_of_features, static_cast<size_t>(number_of_pyramid_levels),
                                                               static_cast<Precision>(scale_factor));
  orb_feature_extractor::Keypoints keypoints;
  feature_extractor.extract(image, keypoints);

  cv::drawKeypoints(image, keypoints, debug_image, cv::Scalar(0, 255, 0));
  cv::imshow("debug_image", debug_image);
  cv::waitKey(0);
}

}  // namespace orb_feature_extractor
