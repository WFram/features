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
  // TODO: back to the loop
  std::map<std::string, std::any> feature_extractor_parameters;
  for (auto &[key, value] : feature_extractor_config.as<std::map<std::string, YAML::Node>>()) {
    feature_extractor_parameters[key] = value.as<std::string>();
  }

  int number_of_features{1500};
  int number_of_pyramid_levels{8};
  Precision scale_factor{1.2};

  std::function read_parameter = [&](const std::string &parameter_name, void *parameter) {
    auto str_value = std::any_cast<std::string>(feature_extractor_parameters.at(parameter_name));
    std::stringstream stream(str_value);
    stream >> parameter;
  };

  read_parameter("number_of_features", &number_of_features);
  std::cout << number_of_features << std::endl;

  cv::Mat image = cv::imread(test_file_, cv::IMREAD_GRAYSCALE);
  cv::Mat debug_image;

  // TODO: solve casting issues
  orb_feature_extractor::ORBFeatureExtractor feature_extractor(
      number_of_features, static_cast<size_t>(number_of_pyramid_levels), static_cast<Precision>(scale_factor));
  orb_feature_extractor::Keypoints keypoints;
  feature_extractor.extract(image, keypoints);

  cv::drawKeypoints(image, keypoints, debug_image, cv::Scalar(0, 255, 0));
  cv::imshow("debug_image", debug_image);
  cv::waitKey(0);
}

}  // namespace orb_feature_extractor
