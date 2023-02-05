set(CONTRIB_URL https://github.com/opencv/opencv_contrib.git)
set(CONTRIB_VERSION 3b5a55876fe0502418a9d9fb7d388c40f2a626b1)

set(contrib_INCLUDE_DIR
        ${CMAKE_CURRENT_BINARY_DIR}/3rd_party/src/opencv_contrib_external/)
set(contrib_LIBS
        ${CMAKE_CURRENT_BINARY_DIR}/3rd_party/src/opencv_contrib_external/modules)

if(NOT EXISTS ${contrib_LIBS})
    ExternalProject_Add(
            opencv_contrib_external
            GIT_REPOSITORY ${CONTRIB_URL}
            GIT_TAG ${CONTRIB_VERSION}
            CONFIGURE_COMMAND ""
            CMAKE_COMMAND ""
            BUILD_COMMAND ""
            INSTALL_COMMAND ""
            TEST_COMMAND ""
            PREFIX 3rd_party
            EXCLUDE_FROM_ALL 1)
    file(MAKE_DIRECTORY ${contrib_INCLUDE_DIR})
endif()

add_library(opencv_contrib INTERFACE IMPORTED GLOBAL)
add_dependencies(opencv_contrib opencv_contrib_external)

set(URL https://github.com/opencv/opencv.git)
set(VERSION 9aa647068b2eba4a34462927b1878353dfd3df69)

set(opencv_INCLUDE_DIR
        ${CMAKE_CURRENT_BINARY_DIR}/3rd_party/src/opencv_external-build/install/include/opencv4
        )
set(opencv_LIBS
        ${CMAKE_CURRENT_BINARY_DIR}/3rd_party/src/opencv_external-build/install/lib)
set(OpenCV_DIR ${CMAKE_CURRENT_BINARY_DIR}/3rd_party/src/opencv_external-build)

if(NOT EXISTS ${opencv_LIBS})
    ExternalProject_Add(
            opencv_external
            DEPENDS opencv_contrib
            GIT_REPOSITORY ${URL}
            GIT_TAG ${VERSION}
            CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_BUILD_TYPE=Release
            -DINSTALL_C_EXAMPLES=OFF
            -DINSTALL_PYTHON_EXAMPLES=OFF
            -DOPENCV_GENERATE_PKGCONFIG=OFF
            -DOPENCV_EXTRA_MODULES_PATH=${contrib_LIBS}
            -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
            -DOPENCV_ENABLE_NONFREE=ON
            -DCMAKE_CXX_STANDARD=20
            -DCMAKE_CXX_STANDARD_REQUIRED=ON
            -DCMAKE_INSTALL_PREFIX=./install
            -DWITH_FFMPEG=ON
            -DBUILD_PERF_TESTS=OFF
            -DBUILD_TESTS=OFF
            -DBUILD_opencv_xphoto=OFF
            -DBUILD_opencv_datasets=OFF
            -DBUILD_opencv_rgbd=OFF
            -DBUILD_opencv_barcode=OFF
            -DBUILD_opencv_xobjdetect=OFF
            -DBUILD_opencv_aruco=OFF
            -DBUILD_opencv_wechat_qrcode=OFF
            -DBUILD_opencv_python3=OFF
            -DBUILD_opencv_python_bindings_generator=OFF
            -DBUILD_opencv_python_tests=OFF
            -DBUILD_opencv_structured_light=OFF
            -DBUILD_opencv_dnn=OFF
            -DBUILD_opencv_dnn_objdetect=OFF
            -DBUILD_opencv_dnn_superres=OFF
            -DBUILD_opencv_face=OFF
            -DBUILD_opencv_bioinspired=OFF
            -DBUILD_opencv_phase_unwrapping=OFF
            -DBUILD_opencv_line_descriptor=OFF
            -DBUILD_opencv_gapi=OFF
            -DBUILD_opencv_intensity_transform=OFF
            -DBUILD_opencv_ml=OFF
            -DBUILD_opencv_photo=OFF
            -DBUILD_opencv_plot=OFF
            -DBUILD_opencv_quality=OFF
            -DBUILD_opencv_reg=OFF
            -DBUILD_opencv_surface_matching=OFF
            -DBUILD_opencv_alphamat=OFF
            -DBUILD_opencv_fuzzy=OFF
            -DBUILD_opencv_hfs=OFF
            -DBUILD_opencv_img_hash=OFF
            -DBUILD_opencv_saliency=OFF
            -DBUILD_opencv_objdetect=OFF
            -DBUILD_opencv_rapid=OFF
            -DBUILD_opencv_shape=OFF
            -DBUILD_opencv_videostab=OFF
            -DBUILD_opencv_ximgproc=OFF
            -DBUILD_opencv_bgsegm=OFF
            -DBUILD_opencv_ccalib=OFF
            -DBUILD_opencv_dpm=OFF
            -DBUILD_opencv_optflow=OFF
            -DBUILD_opencv_sfm=OFF
            -DBUILD_opencv_stitching=OFF
            -DBUILD_opencv_superres=OFF
            -DBUILD_opencv_tracking=OFF
            -DBUILD_opencv_stereo=OFF
            -DBUILD_opencv_hdf=OFF
            -DBUILD_opencv_freetype=OFF
            TEST_COMMAND ""
            PREFIX 3rd_party
            EXCLUDE_FROM_ALL 1)
endif()

add_library(opencv INTERFACE IMPORTED GLOBAL)
add_dependencies(opencv opencv_external)
file(MAKE_DIRECTORY ${opencv_INCLUDE_DIR})
target_include_directories(opencv INTERFACE ${opencv_INCLUDE_DIR})

set(OPENCV_LIBS)

foreach(
        library
        opencv_core
        opencv_flann
        opencv_imgproc
        opencv_features2d
        opencv_imgcodecs
        opencv_videoio
        opencv_calib3d
        opencv_highgui
        opencv_video
        opencv_xfeatures2d)
    list(APPEND OPENCV_LIBS
            ${opencv_LIBS}/lib${library}${CMAKE_SHARED_LIBRARY_SUFFIX})
endforeach()

target_link_libraries(opencv INTERFACE ${OPENCV_LIBS})
