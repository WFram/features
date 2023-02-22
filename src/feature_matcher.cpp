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
  int dist = 0;

  for (int i = 0; i < 8; i++, ptr_descriptor1++, ptr_descriptor2++) {
    int v = *ptr_descriptor1 ^ *ptr_descriptor2;
    v = v - ((v >> 1) & 0x55555555);
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
    dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
  }

  return dist;
}

// Input: keypoints and descriptors
// Output: matches (map between indices of two keypoints vector)
void feature_matcher::BFFeaturematcher::match(const std::vector<cv::KeyPoint> &keypoints1, const cv::Mat &descriptors1,
                                              const std::vector<cv::KeyPoint> &keypoints2, const cv::Mat &descriptors2,
                                              FeatureCorrespondences &correspondences) {
  // TODO: we can consider some filtration here

  // TODO: solve how to use an image pyramid here
  const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

  //Assign keypoints to row table
  vector<vector<size_t> > vRowIndices(nRows, vector<size_t>());

  for (int i = 0; i < nRows; i++)
    vRowIndices[i].reserve(200);

  const int Nr = mvKeysRight.size();

  for (size_t idx1 = 0; idx1 != keypoints1.size(); ++idx1) {
    const auto &kpt1 = keypoints1[idx1];

    const int &level_kpt1 = kpt1.octave;
    const float &vL = kpL.pt.y;
    const float &uL = kpL.pt.x;

    const auto &desc1 = descriptors1.row(static_cast<int>(idx1));



    for (size_t idx2 = 0; idx2 != keypoints2.size(); ++idx2) {
      const auto &kpt2 = keypoints2[idx2];
      const auto &desc2 = descriptors2.row(static_cast<int>(idx2));

      const int distance = descriptorDistance(desc1, desc2);
    }
  }
}

}  // namespace feature_matcher