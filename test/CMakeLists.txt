add_compile_definitions(TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/test_data/")

include_directories("${CMAKE_SOURCE_DIR}/include")
project_add_test(test_orb_feature_extractor test_orb_feature_extractor.cpp feature_extractor yaml-cpp)
project_add_test(test_bf_feature_matcher test_bf_feature_matcher.cpp feature_extractor)
