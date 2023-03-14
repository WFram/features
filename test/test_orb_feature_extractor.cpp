#include <iostream>
#include <vector>

#include <opencv4/opencv2/core.hpp>

#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>
#include <fstream>

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

// TODO: test detector and extractor separately
TEST_F(TestORBFeatureExtractor, testORBFeatureExtractor) {
  auto feature_extractor_config = config_node_["feature_extractor"];
  std::map<std::string, std::string> feature_extractor_parameters;
  for (auto &[key, value] : feature_extractor_config.as<std::map<std::string, YAML::Node>>()) {
    feature_extractor_parameters[key] = value.as<std::string>();
  }

  int number_of_features{1500};
  size_t number_of_pyramid_levels{8};
  Precision scale_factor{1.2f};
  int edge_threshold{19};

  auto read_parameter = [&]<typename T>(const std::string &parameter_name, T &parameter) {
    auto str_value = feature_extractor_parameters.at(parameter_name);
    std::stringstream stream(str_value);
    stream >> parameter;
  };

  read_parameter("number_of_features", number_of_features);
  read_parameter("number_of_pyramid_levels", number_of_pyramid_levels);
  read_parameter("scale_factor", scale_factor);

  cv::Mat image = cv::imread(test_file_, cv::IMREAD_GRAYSCALE);
  cv::Mat debug_image;

  const auto& upscale_vector = utils::computeUpScaleVector(number_of_pyramid_levels, scale_factor);
  const auto& image_pyramid = utils::computeImagePyramid(image, upscale_vector, edge_threshold);

  orb_feature_extractor::ORBFeatureExtractor feature_extractor(number_of_features, upscale_vector);
  orb_feature_extractor::Keypoints keypoints;
  cv::Mat descriptors;
  feature_extractor.extract(image_pyramid, keypoints, descriptors);

  std::cout << "Size of keypoint vector: " << keypoints.size() << "\n";
  std::ofstream debug_log("log.txt", std::ios::trunc);
  for (auto &kp : keypoints) {
    auto projection = kp.pt;
    debug_log << projection << "\n";
  }

  EXPECT_NE(keypoints.size(), 0);
  EXPECT_NE(cv::countNonZero(descriptors), 0);

  cv::drawKeypoints(image, keypoints, debug_image, cv::Scalar(0, 255, 0));
  cv::imshow("debug_image", debug_image);
  cv::waitKey(0);
}

}  // namespace orb_feature_extractor
