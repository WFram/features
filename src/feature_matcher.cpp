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
  const int average_threshold = (high_threshold_ + low_threshold_) / 2;

  // Assign keypoints to a row table
  // TODO: consider replacement to std::unordered_map<size_t, size_t>
  std::vector<std::vector<size_t>> row_indices(static_cast<size_t>(imagePyramid1_[0].rows), std::vector<size_t>());

  // TODO: if 200 is a constant parameter, find out the variable
  for (size_t i = 0; i < static_cast<size_t>(imagePyramid1_[0].rows); i++) row_indices[i].reserve(200);

  for (size_t idx_2 = 0; idx_2 < keypoints2.size(); idx_2++) {
    const cv::KeyPoint &kp2 = keypoints2[idx_2];
    // TODO: solve how to deal with different possible precisions here
    const float double_scale = 2.0f * scale_factor_per_level_[static_cast<size_t>(keypoints2[idx_2].octave)];

    const int y_high_boarder = static_cast<int>(std::ceil(kp2.pt.y + double_scale));
    const int y_low_boarder = static_cast<int>(std::floor(kp2.pt.y - double_scale));

    // row_indices contains the correspondences between an index of a keypoints in the 2nd image and its y coordinate
    // TODO: if so, let's do as a map
    for (int yi = y_low_boarder; yi <= y_high_boarder; yi++) row_indices[static_cast<size_t>(yi)].push_back(idx_2);
  }

  /*std::vector<std::pair<int, int>> distance_indices;
    distance_indices.reserve(keypoints1.size());*/

  // For each keypoint from the 1st image search a match in the 2nd image
  for (size_t idx_1 = 0; idx_1 < keypoints1.size(); idx_1++) {
    const cv::KeyPoint &kp1 = keypoints1[idx_1];
    const int &level_kp1 = kp1.octave;

    // TODO: consider conversion to an index more carefully
    const std::vector<size_t> &match_candidates = row_indices[static_cast<size_t>(kp1.pt.y)];

    if (match_candidates.empty()) continue;

    int best_distance = high_threshold_;
    size_t best_candidate_index = 0;

    const cv::Mat &desc1 = descriptors1.row(static_cast<int>(idx_1));

    // Compare descriptor to right keypoints
    for (size_t idx_cand = 0; idx_cand < match_candidates.size(); idx_cand++) {
      const size_t idx_2 = match_candidates[idx_cand];
      const cv::KeyPoint &kp2 = keypoints2[idx_2];

      if (kp2.octave < level_kp1 - 1 || kp2.octave > level_kp1 + 1) continue;

      if (std::fabs(kp2.pt.x - kp1.pt.x) <= 1e-5) {
        const cv::Mat &desc_2 = descriptors2.row(static_cast<int>(idx_2));
        const int distance = descriptorDistance(desc1, desc_2);

        if (distance < best_distance) {
          best_distance = distance;
          best_candidate_index = idx_2;
        }
      }
    }

    correspondences[idx_1] = best_candidate_index;

    // TODO: decide if we need this additional refinement

    /*// TODO: make in a separate function
    // Subpixel match by correlation
    if (best_distance < average_threshold) {
      // Coordinates in an image pyramid at keypoint scale
      const float kp1_inv_scale_factor = inv_scale_factor_per_level_[static_cast<size_t>(kp1.octave)];
      const int scaled_kpt1_x = static_cast<int>(std::round(kp1.pt.x * kp1_inv_scale_factor));
      const int scaled_kpt1_y = static_cast<int>(std::round(kp1.pt.y * kp1_inv_scale_factor));
      const int scaled_best_kp2_x =
          static_cast<int>(std::round(keypoints2[best_candidate_index].pt.x * kp1_inv_scale_factor));

      // Sliding window search
      // TODO: make configurable
      const int window_size = 5;
      cv::Mat image_1_cropped = imagePyramid1_[static_cast<size_t>(kp1.octave)]
                                    .rowRange(scaled_kpt1_y - window_size, scaled_kpt1_y + window_size + 1)
                                    .colRange(scaled_kpt1_x - window_size, scaled_kpt1_x + window_size + 1);

      best_distance = std::numeric_limits<int>::max();
      int best_increment = 0;
      const int shift = 5;
      std::vector<float> subpixel_distances;
      subpixel_distances.resize(2 * shift + 1);

      const int left_shifted_best_kp2_x = scaled_best_kp2_x + shift - window_size;
      const int right_shifted_best_kp2_x = scaled_best_kp2_x + shift + window_size + 1;
      if (left_shifted_best_kp2_x < 0 ||
          right_shifted_best_kp2_x >= imagePyramid2_[static_cast<size_t>(kp1.octave)].cols)
        continue;

      for (int column_increment = -shift; column_increment <= +shift; column_increment++) {
        cv::Mat image_2_cropped = imagePyramid2_[static_cast<size_t>(kp1.octave)]
                                      .rowRange(scaled_kpt1_y - window_size, scaled_kpt1_y + window_size + 1)
                                      .colRange(scaled_best_kp2_x + column_increment - window_size,
                                                scaled_best_kp2_x + column_increment + window_size + 1);

        float subpixel_distance = static_cast<Precision>(cv::norm(image_1_cropped, image_2_cropped, cv::NORM_L1));
        if (subpixel_distance < static_cast<Precision>(best_distance)) {
          // TODO: it's rough. Don't lose precision here
          best_distance = static_cast<int>(subpixel_distance);
          best_increment = column_increment;
        }

        subpixel_distances[static_cast<size_t>(shift + column_increment)] = subpixel_distance;
      }

      if (best_increment == -shift || best_increment == shift) continue;

      // Sub-pixel match (Parabola fitting)
      const float subpixel_distance_left = subpixel_distances[static_cast<size_t>(shift + best_increment - 1)];
      const float subpixel_distance_central = subpixel_distances[static_cast<size_t>(shift + best_increment)];
      const float subpixel_distance_right = subpixel_distances[static_cast<size_t>(shift + best_increment + 1)];

      const float subpixel_increment =
          (subpixel_distance_left - subpixel_distance_right) /
          (2.0f * (subpixel_distance_left + subpixel_distance_right - 2.0f * subpixel_distance_central));

      if (subpixel_increment < -1 || subpixel_increment > 1) continue;

      // Re-scaled coordinate
      float rescaled_best_kp2_x =
          scale_factor_per_level_[static_cast<size_t>(kp1.octave)] *
          (static_cast<float>(scaled_best_kp2_x) + static_cast<float>(best_increment) + subpixel_increment);

      float disparity = (kp1.pt.x - rescaled_best_kp2_x);

      if (disparity <= 0) {
        disparity = 0.01f;
        rescaled_best_kp2_x = kp1.pt.x - 0.01f;
      }
      distance_indices.push_back(std::pair<int, int>(best_distance, static_cast<int>(idx_1)));
    }*/
  }

  /*std::sort(distance_indices.begin(), distance_indices.end());*/
}

}  // namespace feature_matcher