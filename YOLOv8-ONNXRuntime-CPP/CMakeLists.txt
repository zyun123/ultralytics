cmake_minimum_required(VERSION 3.5)

set(PROJECT_NAME Yolov8OnnxRuntimeCPPInference)
project(${PROJECT_NAME} VERSION 0.0.1 LANGUAGES CXX)


# -------------- Support C++17 for using filesystem  ------------------#
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)


# -------------- OpenCV  ------------------#
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


# -------------- Compile CUDA for FP16 inference if needed  ------------------#
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})


# -------------- ONNXRUNTIME  ------------------#



# Set ONNXRUNTIME_VERSION
set(ONNXRUNTIME_VERSION 1.10.0)
set(ONNXRUNTIME_ROOT "/home/zy/Downloads/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}")
message(STATUS "use onnxruntime gpu")
message(STATUS "onnxruntime-gpu dir: ${ONNXRUNTIME_ROOT}")



include_directories(${PROJECT_NAME} ${ONNXRUNTIME_ROOT}/include)


add_executable(${PROJECT_NAME} custom_inference.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so)
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})


