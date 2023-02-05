cmake_minimum_required(VERSION 3.21)
project(wf_feature_extractor C CXX)

set(CMAKE_CXX_STANDARD 20)

include(dependencies.cmake)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 11.0)
        # wouldn't build on g++-12 without this
        set(CMAKE_CXX_FLAGS
                "${CMAKE_CXX_FLAGS} -Wno-maybe-uninitialized -Wno-uninitialized -Wno-restrict -Wno-array-bounds -Wno-strict-overflow -Wno-aggressive-loop-optimizations"
                )
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wstrict-overflow=2")
    endif()
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -pthread")
set(CMAKE_CXX_FLAGS
        "${CMAKE_CXX_FLAGS} -Werror -Wall -Wextra -Wpedantic -Wconversion -fdiagnostics-color=always"
        )

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS
            "${CMAKE_CXX_FLAGS} -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization"
            )
    set(CMAKE_CXX_FLAGS
            "${CMAKE_CXX_FLAGS} -Wformat=2 -Winit-self -Wlogical-op -Wmissing-include-dirs -Wnoexcept"
            )
    set(CMAKE_CXX_FLAGS
            "${CMAKE_CXX_FLAGS} -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow"
            )
    set(CMAKE_CXX_FLAGS
            "${CMAKE_CXX_FLAGS} -Wsign-promo -Wsign-conversion -Wstrict-null-sentinel"
            )
    set(CMAKE_CXX_FLAGS
            "${CMAKE_CXX_FLAGS} -Wundef -Wunused-variable -Wno-variadic-macros -Wno-parentheses"
            )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconcepts-diagnostics-depth=2")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -no-pie")
endif()

add_library(feature_extractor
            src/feature_extractor.cpp)

target_include_directories(feature_extractor PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(feature_extractor PUBLIC opencv)

add_executable(wf_feature_extractor main.cpp)
target_link_libraries(wf_feature_extractor feature_extractor)
