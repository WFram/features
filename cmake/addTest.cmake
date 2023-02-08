# Wrapper to simplify test
function(project_add_test test_name test_source)
  add_executable(${test_name} ${test_source})
  target_link_libraries(${test_name} gtest gtest_main)
  target_link_libraries(${test_name} ${ARGN})
  add_test(${test_name} ${test_name})
endfunction()
