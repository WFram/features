//
// Created by wfram on 2/22/23.
//

#ifndef WF_FEATURE_EXTRACTOR_BENCHMARK_H
#define WF_FEATURE_EXTRACTOR_BENCHMARK_H

#include <feature_extractor.h>
#include <feature_matcher.h>

namespace benchmark {

class Benchmark {
 public:
  explicit Benchmark() {
    feature_extractor_ = std::make_unique<orb_feature_extractor::ORBFeatureExtractor>(
        number_of_features_, number_of_pyramid_levels_, scale_factor);
  }

 private:
  // TODO: add config loader here

  std::unique_ptr<orb_feature_extractor::ORBFeatureExtractor> feature_extractor_;
  std::unique_ptr<feature_matcher::FeatureMatcher> feature_matcher_;

  int number_of_features_ = 1500;
  size_t number_of_pyramid_levels_ = 8;
  orb_feature_extractor::Precision scale_factor = 1.2;
};

}  // namespace benchmark

#endif  // WF_FEATURE_EXTRACTOR_BENCHMARK_H
