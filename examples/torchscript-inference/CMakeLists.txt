cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(script)
set(Torch_DIR /home/zy/Downloads/libtorch-cxx11-abi-shared-with-deps-1.9.1+cu111/libtorch/share/cmake/Torch)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(TorchVision REQUIRED)
# set(PROJECT_SOURCES
#     main.cpp
#     detect_hand_foot.h
#     detect_hand_foot.cpp
# )



add_executable(torchscript main.cpp detect_hand_foot.cpp detect_hand_foot.h)
# add_library(torchscript SHARED  detect_hand_foot.cpp detect_hand_foot.h)

target_link_libraries(torchscript "${TORCH_LIBRARIES}" "${OpenCV_LIBRARIES}" TorchVision::TorchVision stdc++fs)


set_property(TARGET torchscript PROPERTY CXX_STANDARD 14 )