//
// Created by wfram on 1/28/23.
//

#include "feature_extractor.h"

namespace orb_feature_extractor {
ORBFeatureExtractor::ORBFeatureExtractor(const int &number_of_features, const std::vector<Precision> &upscale_vector)
    : number_of_features_(number_of_features), scale_factor_(upscale_vector[1]), upscale_vector_(upscale_vector) {
  const int number_of_keypoints(512);
  // TODO: check, if it's converted correctly
  // TODO: change an array to std::array
  const auto *pattern = reinterpret_cast<const cv::Point *>(bit_pattern_31_);
  std::copy(pattern, pattern + number_of_keypoints, std::back_inserter(pattern_));

  umax_.resize(static_cast<size_t>(half_patch_size_ + 1));

  int v, v0, vmax = cvFloor(static_cast<Precision>(half_patch_size_) * std::sqrt(2.0) / 2 + 1);
  int vmin = cvCeil(static_cast<Precision>(half_patch_size_) * sqrt(2.0) / 2);
  const auto squared_half_patch_size_ = static_cast<Precision>(half_patch_size_ * half_patch_size_);
  for (v = 0; v <= vmax; ++v)
    umax_[static_cast<size_t>(v)] = cvRound(std::sqrt(squared_half_patch_size_ - static_cast<Precision>((v * v))));

  // Make sure about symmetry
  for (v = half_patch_size_, v0 = 0; v >= vmin; --v) {
    while (umax_[static_cast<size_t>(v0)] == umax_[static_cast<size_t>(v0 + 1)]) ++v0;
    umax_[static_cast<size_t>(v)] = v0;
    ++v0;
  }
}

void ORBFeatureExtractor::ExtractorNode::divideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3,
                                                    ExtractorNode &n4) {
  const int half_x = static_cast<int>(std::ceil(static_cast<Precision>(u_r_.x - u_l_.x) / 2));
  const int half_y = static_cast<int>(std::ceil(static_cast<Precision>(b_r_.y - u_l_.y) / 2));

  // Define boundaries of children
  std::function define_boundaries = [&](ExtractorNode &n, cv::Point2i u_l, cv::Point2i u_r, cv::Point2i b_l,
                                        cv::Point2i b_r) {
    n.u_l_ = u_l;
    n.u_r_ = u_r;
    n.b_l_ = b_l;
    n.b_r_ = b_r;
    n.keypoints_.reserve(keypoints_.size());
  };

  define_boundaries(n1, u_l_, cv::Point2i(u_l_.x + half_x, u_l_.y), cv::Point2i(u_l_.x, u_l_.y + half_y),
                    cv::Point2i(u_l_.x + half_x, u_l_.y + half_y));
  define_boundaries(n2, n1.u_r_, u_r_, n1.b_r_, cv::Point2i(u_r_.x, u_l_.y + half_y));
  define_boundaries(n3, n1.b_l_, n1.b_r_, b_l_, cv::Point2i(n1.b_r_.x, b_l_.y));
  define_boundaries(n4, n3.u_r_, n2.b_r_, n3.b_r_, b_r_);

  // Associate points to children
  for (auto &kp : keypoints_) {
    if (kp.pt.x < static_cast<float>(n1.u_r_.x)) {
      if (kp.pt.y < static_cast<float>(n1.b_r_.y))
        n1.keypoints_.push_back(kp);
      else
        n3.keypoints_.push_back(kp);
    } else if (kp.pt.y < static_cast<float>(n1.b_r_.y))
      n2.keypoints_.push_back(kp);
    else
      n4.keypoints_.push_back(kp);
  }

  if (n1.keypoints_.size() == 1) n1.no_more_ = true;
  if (n2.keypoints_.size() == 1) n2.no_more_ = true;
  if (n3.keypoints_.size() == 1) n3.no_more_ = true;
  if (n4.keypoints_.size() == 1) n4.no_more_ = true;
}

bool ORBFeatureExtractor::ExtractorNode::compareNodes(std::pair<int, ExtractorNode *> &e1,
                                                      std::pair<int, ExtractorNode *> &e2) {
  if (e1.first < e2.first) {
    return true;
  } else if (e1.first > e2.first) {
    return false;
  } else {
    if (e1.second->u_l_.x < e2.second->u_l_.x) {
      return true;
    } else {
      return false;
    }
  }
}

void ORBFeatureExtractor::distributeOctTree(const Keypoints &distributed_keypoints, Keypoints &result_keypoints,
                                            const int &min_x, const int &max_x, const int &min_y, const int &max_y,
                                            const size_t &N) const {
  // Compute how many initial nodes are there
  const int init_n = static_cast<int>(std::round((max_x - min_x) / (max_y - min_y)));
  // TODO: rename
  const Precision h_x = static_cast<Precision>(max_x - min_x) / static_cast<Precision>(init_n);

  std::list<ExtractorNode> nodes;
  std::vector<ExtractorNode *> init_nodes;
  init_nodes.resize(static_cast<size_t>(init_n));

  for (int i = 0; i < init_n; i++) {
    ExtractorNode ni;
    ni.u_l_ = cv::Point2i(static_cast<int>(h_x * static_cast<Precision>(i)), 0);
    ni.u_r_ = cv::Point2i(static_cast<int>(h_x * static_cast<float>(i + 1)), 0);
    ni.b_l_ = cv::Point2i(ni.u_l_.x, max_y - min_y);
    ni.b_r_ = cv::Point2i(ni.u_r_.x, max_y - min_y);
    ni.keypoints_.reserve(distributed_keypoints.size());

    nodes.push_back(ni);
    init_nodes[static_cast<size_t>(i)] = &nodes.back();
  }

  // Associate points to children
  for (const auto &kp : distributed_keypoints) init_nodes[static_cast<size_t>(kp.pt.x / h_x)]->keypoints_.push_back(kp);

  auto it = nodes.begin();

  while (it != nodes.end()) {
    if (it->keypoints_.size() == 1) {
      it->no_more_ = true;
      it++;
    } else if (it->keypoints_.empty())
      it = nodes.erase(it);
    else
      it++;
  }

  bool finish{false};
  int iteration{0};

  std::vector<std::pair<int, ExtractorNode *> > size_node_pairs;
  size_node_pairs.reserve(nodes.size() * 4);

  std::function add_children = [&](ExtractorNode &n, std::list<ExtractorNode> &tree_nodes,
                                   std::vector<std::pair<int, ExtractorNode *> > &pairs, int &to_expand) {
    if (!n.keypoints_.empty()) {
      tree_nodes.push_front(n);
      if (n.keypoints_.size() > 1) {
        to_expand++;
        pairs.emplace_back(n.keypoints_.size(), &tree_nodes.front());
        tree_nodes.front().node_iterator_ = tree_nodes.begin();
      }
    }
  };

  while (!finish) {
    iteration++;
    size_t previous_size{nodes.size()};
    it = nodes.begin();
    int num_to_expand{0};
    size_node_pairs.clear();

    while (it != nodes.end()) {
      if (it->no_more_) {
        // If node only contains one point, do not subdivide and continue
        it++;
        continue;
      } else {
        // If more than one point, subdivide
        // TODO: change them to pointers
        ExtractorNode n1, n2, n3, n4;
        it->divideNode(n1, n2, n3, n4);

        // Add children if they contain points

        // TODO: an array on nodes
        add_children(n1, nodes, size_node_pairs, num_to_expand);
        add_children(n2, nodes, size_node_pairs, num_to_expand);
        add_children(n3, nodes, size_node_pairs, num_to_expand);
        add_children(n4, nodes, size_node_pairs, num_to_expand);

        it = nodes.erase(it);
        continue;
      }
    }

    // Finish if there are more nodes than required features or all nodes contain just one point
    if (nodes.size() >= N || nodes.size() == previous_size) {
      finish = true;
    } else if ((nodes.size() + static_cast<size_t>(num_to_expand * 3)) > N) {
      while (!finish) {
        previous_size = nodes.size();

        std::vector<std::pair<int, ExtractorNode *> > previous_size_node_pairs = size_node_pairs;
        size_node_pairs.clear();

        std::sort(previous_size_node_pairs.begin(), previous_size_node_pairs.end(), ExtractorNode::compareNodes);
        // TODO: check it it's not 0 here
        for (size_t j = previous_size_node_pairs.size() - 1; j--;) {
          ExtractorNode n1, n2, n3, n4;
          previous_size_node_pairs[j].second->divideNode(n1, n2, n3, n4);

          // Add children if they contain points
          add_children(n1, nodes, size_node_pairs, num_to_expand);
          add_children(n2, nodes, size_node_pairs, num_to_expand);
          add_children(n3, nodes, size_node_pairs, num_to_expand);
          add_children(n4, nodes, size_node_pairs, num_to_expand);

          nodes.erase(previous_size_node_pairs[j].second->node_iterator_);

          if (nodes.size() >= N) break;
        }

        if (nodes.size() >= N || nodes.size() == previous_size) finish = true;
      }
    }
  }

  // Retain the best point in each node
  for (auto &n : nodes) {
    Keypoints &node_keypoints = n.keypoints_;
    cv::KeyPoint *keypoint = &node_keypoints[0];
    Precision max_response = keypoint->response;

    for (size_t k = 1; k < node_keypoints.size(); k++) {
      if (node_keypoints[k].response > max_response) {
        keypoint = &node_keypoints[k];
        max_response = node_keypoints[k].response;
      }
    }

    result_keypoints.push_back(*keypoint);
  }
}

float ORBFeatureExtractor::ICAngle(const cv::Mat &image, cv::Point2f pt, const std::vector<int> &u_max) const {
  float m_01{0.0}, m_10{0.0};
  const uchar *center = &image.at<uchar>(cvRound(pt.y), cvRound(pt.x));

  // Treat the center line differently, v = 0
  for (int u = -half_patch_size_; u <= half_patch_size_; ++u) m_10 += static_cast<float>(u * center[u]);

  // Go line by line in the circuI853lar patch
  int step = static_cast<int>(image.step1());
  for (int v = 1; v <= half_patch_size_; ++v) {
    // Proceed over the two lines
    int v_sum{0};
    int d{u_max[static_cast<size_t>(v)]};
    for (int u = -d; u <= d; ++u) {
      int val_plus = center[u + v * step], val_minus = center[u - v * step];
      v_sum += (val_plus - val_minus);
      m_10 += static_cast<float>(u * (val_plus + val_minus));
    }
    m_01 += static_cast<float>(v * v_sum);
  }

  return cv::fastAtan2(m_01, m_10);
}

void ORBFeatureExtractor::computeOrientation(const cv::Mat &image, Keypoints &keypoints,
                                             const std::vector<int> &u_max) const {
  for (auto &kp : keypoints) kp.angle = ICAngle(image, kp.pt, u_max);
}

void ORBFeatureExtractor::computeKeypointsOctTree(std::vector<Keypoints> &all_keypoints) const {
  all_keypoints.resize(image_pyramid_.size());
  const int window_size(35);

  for (size_t level = 0; level < image_pyramid_.size(); ++level) {
    // TODO: think of setting them as class members
    const int min_border_x(edge_threshold_ - 3);
    const int min_border_y(min_border_x);
    const int max_border_x(image_pyramid_.at(level).cols - edge_threshold_ + 3);
    const int max_border_y(image_pyramid_.at(level).rows - edge_threshold_ + 3);

    Keypoints distributed_keypoints;
    distributed_keypoints.reserve(static_cast<size_t>(number_of_features_ * 10));

    const int width(max_border_x - min_border_x);
    const int height(max_border_y - min_border_y);

    const int cols(width / window_size);
    const int rows(height / window_size);
    const int cell_width(static_cast<int>(std::ceil(width / cols)));
    const int cell_height(static_cast<int>(std::ceil(height / rows)));

    for (int i = 0; i < rows; i++) {
      const int init_y = min_border_y + i * cell_height;
      int max_y = init_y + cell_height + 6;

      if (init_y >= max_border_y - 3) continue;
      if (max_y > max_border_y) max_y = max_border_y;

      for (int j = 0; j < cols; j++) {
        const int init_x = min_border_x + j * cell_width;
        int max_x = init_x + cell_width + 6;
        if (init_x >= max_border_x - 6) continue;
        if (max_x > max_border_x) max_x = max_border_x;

        Keypoints cell_keypoints;
        // TODO: change to flexible version
        cv::FAST(image_pyramid_.at(level).rowRange(init_y, max_y).colRange(init_x, max_x), cell_keypoints,
                 init_fast_threshold_, true);

        if (cell_keypoints.empty())
          cv::FAST(image_pyramid_.at(level).rowRange(init_y, max_y).colRange(init_x, max_x), cell_keypoints,
                   min_fast_threshold_, true);

        // TODO: check if we don't lose the precision when casting
        if (!cell_keypoints.empty()) {
          for (auto &cell_kp : cell_keypoints) {
            cell_kp.pt.x += static_cast<float>(j * cell_width);
            cell_kp.pt.y += static_cast<float>(i * cell_height);
            distributed_keypoints.push_back(cell_kp);
          }
        }
      }
    }

    auto &keypoints = all_keypoints[level];
    keypoints.reserve(static_cast<size_t>(number_of_features_));

    distributeOctTree(distributed_keypoints, keypoints, min_border_x, max_border_x, min_border_y, max_border_y,
                      static_cast<size_t>(features_per_level_[level]));

    const int scaled_patch_size = static_cast<int>(static_cast<Precision>(patch_size_) * upscale_vector_[level]);

    // Add border to the coordinates and scale information
    for (auto &kp : keypoints) {
      kp.pt.x += static_cast<float>(min_border_x);
      kp.pt.y += static_cast<float>(min_border_y);
      kp.octave = static_cast<int>(level);
      kp.size = static_cast<float>(scaled_patch_size);
    }
  }

  for (size_t level = 0; level < image_pyramid_.size(); ++level)
    computeOrientation(image_pyramid_.at(level), all_keypoints[level], umax_);
}

void ORBFeatureExtractor::compute_descriptors(const cv::Mat &image, const Keypoints &keypoints, cv::Mat &descriptors) {
  descriptors = cv::Mat::zeros(static_cast<int>(keypoints.size()), 32, CV_8U);

  for (size_t i = 0; i < keypoints.size(); i++) {
    Precision angle = static_cast<Precision>(keypoints[i].angle) * static_cast<Precision>(CV_PI / 180);

    const uchar *center = &image.at<uchar>(cvRound(keypoints[i].pt.y), cvRound(keypoints[i].pt.x));
    const int image_step = static_cast<int>(image.step);

    auto pattern = &pattern_[0];

    auto get_center_pixel_value = [&](const int &idx) {
      return center[cvRound(static_cast<Precision>(pattern[static_cast<size_t>(idx)].x) * std::sin(angle) +
                            static_cast<Precision>(pattern[static_cast<size_t>(idx)].y) * std::cos(angle)) *
                        image_step +
                    cvRound(static_cast<Precision>(pattern[static_cast<size_t>(idx)].x) * std::cos(angle) -
                            static_cast<Precision>(pattern[static_cast<size_t>(idx)].y) * std::sin(angle))];
    };

    for (int j = 0; j < 32; ++j, pattern += 16) {
      int t0, t1, val;
      t0 = get_center_pixel_value(0);
      t1 = get_center_pixel_value(1);
      val = t0 < t1;
      t0 = get_center_pixel_value(2);
      t1 = get_center_pixel_value(3);
      val |= (t0 < t1) << 1;
      t0 = get_center_pixel_value(4);
      t1 = get_center_pixel_value(5);
      val |= (t0 < t1) << 2;
      t0 = get_center_pixel_value(6);
      t1 = get_center_pixel_value(7);
      val |= (t0 < t1) << 3;
      t0 = get_center_pixel_value(8);
      t1 = get_center_pixel_value(9);
      val |= (t0 < t1) << 4;
      t0 = get_center_pixel_value(10);
      t1 = get_center_pixel_value(11);
      val |= (t0 < t1) << 5;
      t0 = get_center_pixel_value(12);
      t1 = get_center_pixel_value(13);
      val |= (t0 < t1) << 6;
      t0 = get_center_pixel_value(14);
      t1 = get_center_pixel_value(15);
      val |= (t0 < t1) << 7;

      // TODO: think of changing static_cast to some alias
      auto descriptor = descriptors.ptr(static_cast<int>(i));
      descriptor[j] = static_cast<uchar>(val);
    }
  }
}

void ORBFeatureExtractor::extract(const ImagePyramid &image_pyramid, Keypoints &keypoints, cv::Mat &descriptors) {
  if (image_pyramid.at(0).empty()) {
    std::cerr << "ERROR: empty image!" << std::endl;
    return;
  }

  assert(image_pyramid.at(0).type() == CV_8UC1);

  // TODO: use a setter
  image_pyramid_ = image_pyramid;

  // TODO: a separate function
  features_per_level_.resize(image_pyramid_.size());

  Precision factor = 1.0f / scale_factor_;
  Precision desired_features_per_scale =
      static_cast<Precision>(number_of_features_) * (1.0f - factor) /
      (1.0f - static_cast<Precision>(
                  std::pow(static_cast<Precision>(factor), static_cast<Precision>(image_pyramid_.size()))));

  int sum_of_features{0};
  for (size_t level = 0; level < image_pyramid_.size() - 1; level++) {
    features_per_level_[level] = cvRound(desired_features_per_scale);
    sum_of_features += features_per_level_[level];
    desired_features_per_scale *= factor;
  }
  features_per_level_[image_pyramid_.size() - 1] = std::max(number_of_features_ - sum_of_features, 0);

  std::vector<Keypoints> all_keypoints;
  computeKeypointsOctTree(all_keypoints);

  // TODO: shouldn't it be an alias?
  cv::Mat temp_descriptors;

  size_t number_of_keypoints_in_pyramid{0};
  for (size_t level = 0; level < image_pyramid_.size(); ++level)
    number_of_keypoints_in_pyramid += all_keypoints[level].size();

  if (number_of_keypoints_in_pyramid == 0)
    descriptors.release();
  else {
    // TODO: figure out what happens (what are the values after we did it)
    descriptors = cv::Mat::zeros(static_cast<int>(number_of_keypoints_in_pyramid), 32, CV_8U);
    // TODO: make sure everything is okay here
    temp_descriptors = descriptors;
  }

  keypoints = Keypoints(number_of_keypoints_in_pyramid);

  for (size_t level = 0; level < image_pyramid_.size(); ++level) {
    Keypoints &keypoints_per_level = all_keypoints[level];
    size_t number_of_keypoints_per_level = keypoints_per_level.size();
    // TODO: check simply is it empty or not
    if (number_of_keypoints_per_level == 0) continue;

    /* Here we should deal with descriptors */
    auto blurred_image = image_pyramid_.at(level).clone();
    cv::GaussianBlur(blurred_image, blurred_image, cv::Size(7, 7), 2, 2, cv::BORDER_REFLECT_101);

    cv::Mat descriptors_per_level = cv::Mat(static_cast<int>(number_of_keypoints_per_level), 32, CV_8U);
    compute_descriptors(blurred_image, keypoints_per_level, descriptors_per_level);

    // TODO: maybe the index to fill a vector could be the reason
    Precision scale = upscale_vector_[level];
    for (size_t i = 0; i != keypoints_per_level.size(); ++i) {
      // Scale keypoint coordinates
      auto kp_per_lvl = keypoints_per_level[i];
      if (level != 0) kp_per_lvl.pt *= scale;
      keypoints.at(i) = kp_per_lvl;
      descriptors_per_level.row(static_cast<int>(i)).copyTo(temp_descriptors.row(static_cast<int>(i)));
    }
  }
}
}  // namespace orb_feature_extractor
