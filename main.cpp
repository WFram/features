#include <iostream>
#include <vector>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>

#include "include/feature_extractor.h"

// TODO: create an abstract class which performs feature extraction
//          it should provide at least two inherited feature extractors: SIFT and ORB
//          ORB extractor should be written from the scratch
//          for SIFT the standard OpenCV implementation can be used (for now)
//          when an instance is created it should pass a type of the feature extractor
//          to extract the features it also should have an ability to specify if we need grayscale or color image

int main() {
  // TODO: from test folder
  std::string image_path = "/home/wfram/wf_feature_extractor/test/00000.jpg";
  cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
  cv::Mat debug_image;

  // TODO: set from YAML file
  orb_feature_extractor::ORBFeatureExtractor feature_extractor(1500, 8, 1.2);
  orb_feature_extractor::Keypoints keypoints;
  feature_extractor.extract(image, keypoints);

  cv::drawKeypoints(image, keypoints, debug_image);
  cv::waitKey(0);

  return 0;
}
