//
// Created by wfram on 2/22/23.
//

#include <feature_matcher.h>

namespace feature_matcher {

// hamming
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
int feature_matcher::BFFeaturematcher::descriptorDistance(const cv::Mat &descriptor1, const cv::Mat &descriptor2) {
  const int *ptr_descriptor1 = descriptor1.ptr<int32_t>();
  const int *ptr_descriptor2 = descriptor2.ptr<int32_t>();
  int distance = 0;

  for (int i = 0; i < 8; i++, ptr_descriptor1++, ptr_descriptor2++) {
    int v = *ptr_descriptor1 ^ *ptr_descriptor2;
    v = v - ((v >> 1) & 0x55555555);
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
    distance += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
  }

  return distance;
}

// Input: keypoints and descriptors
// Output: matches (map between indices of two keypoints vector)
void feature_matcher::BFFeaturematcher::match(const std::vector<cv::KeyPoint> &keypoints1, const cv::Mat &descriptors1,
                                              const std::vector<cv::KeyPoint> &keypoints2, const cv::Mat &descriptors2,
                                              FeatureCorrespondences &correspondences) {
  std::vector<int> distances(keypoints2.size(), INT_MAX);

  for (size_t idx_from = 0, idx_from_end = keypoints1.size(); idx_from < idx_from_end; idx_from++) {
    cv::KeyPoint keypoint = keypoints1[idx_from];
    int level = keypoint.octave;
    if (level > 0) continue;

    cv::Mat descriptor_from = descriptors1.row(static_cast<int>(idx_from));

    int first_best_distance = INT_MAX;
    int second_best_distance = INT_MAX;
    int best_candidate_idx = -1;

    for (size_t idx_to = 0, idx_to_end = keypoints2.size(); idx_to < idx_to_end; idx_to++) {
      cv::Mat descriptor_to = descriptors2.row(static_cast<int>(idx_to));

      int distance = descriptorDistance(descriptor_from, descriptor_to);

      // TODO: here check if the newest distance is less than previously computed

      if (distance < first_best_distance) {
        second_best_distance = first_best_distance;
        first_best_distance = distance;
        best_candidate_idx = static_cast<int>(idx_to);
      } else if (distance < second_best_distance) {
        second_best_distance = distance;
      }
    }

    if (first_best_distance <= low_threshold_) {
      if (static_cast<Precision>(first_best_distance) < static_cast<Precision>(second_best_distance) * lowe_ratio_) {
        correspondences[idx_from] = static_cast<size_t>(best_candidate_idx);
      }
    }
  }
}

}  // namespace feature_matcher