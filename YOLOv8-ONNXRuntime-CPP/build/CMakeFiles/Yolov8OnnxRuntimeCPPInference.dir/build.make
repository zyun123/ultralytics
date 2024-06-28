# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.18

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/zy/tools/cmake-3.18.5-Linux-x86_64/bin/cmake

# The command to remove a file.
RM = /home/zy/tools/cmake-3.18.5-Linux-x86_64/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/zy/vision/ultralytics/examples/YOLOv8-ONNXRuntime-CPP

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zy/vision/ultralytics/examples/YOLOv8-ONNXRuntime-CPP/build

# Include any dependencies generated for this target.
include CMakeFiles/Yolov8OnnxRuntimeCPPInference.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Yolov8OnnxRuntimeCPPInference.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Yolov8OnnxRuntimeCPPInference.dir/flags.make

CMakeFiles/Yolov8OnnxRuntimeCPPInference.dir/custom_inference.cpp.o: CMakeFiles/Yolov8OnnxRuntimeCPPInference.dir/flags.make
CMakeFiles/Yolov8OnnxRuntimeCPPInference.dir/custom_inference.cpp.o: ../custom_inference.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zy/vision/ultralytics/examples/YOLOv8-ONNXRuntime-CPP/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Yolov8OnnxRuntimeCPPInference.dir/custom_inference.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Yolov8OnnxRuntimeCPPInference.dir/custom_inference.cpp.o -c /home/zy/vision/ultralytics/examples/YOLOv8-ONNXRuntime-CPP/custom_inference.cpp

CMakeFiles/Yolov8OnnxRuntimeCPPInference.dir/custom_inference.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Yolov8OnnxRuntimeCPPInference.dir/custom_inference.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zy/vision/ultralytics/examples/YOLOv8-ONNXRuntime-CPP/custom_inference.cpp > CMakeFiles/Yolov8OnnxRuntimeCPPInference.dir/custom_inference.cpp.i

CMakeFiles/Yolov8OnnxRuntimeCPPInference.dir/custom_inference.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Yolov8OnnxRuntimeCPPInference.dir/custom_inference.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zy/vision/ultralytics/examples/YOLOv8-ONNXRuntime-CPP/custom_inference.cpp -o CMakeFiles/Yolov8OnnxRuntimeCPPInference.dir/custom_inference.cpp.s

# Object files for target Yolov8OnnxRuntimeCPPInference
Yolov8OnnxRuntimeCPPInference_OBJECTS = \
"CMakeFiles/Yolov8OnnxRuntimeCPPInference.dir/custom_inference.cpp.o"

# External object files for target Yolov8OnnxRuntimeCPPInference
Yolov8OnnxRuntimeCPPInference_EXTERNAL_OBJECTS =

Yolov8OnnxRuntimeCPPInference: CMakeFiles/Yolov8OnnxRuntimeCPPInference.dir/custom_inference.cpp.o
Yolov8OnnxRuntimeCPPInference: CMakeFiles/Yolov8OnnxRuntimeCPPInference.dir/build.make
Yolov8OnnxRuntimeCPPInference: /usr/local/lib/libopencv_gapi.so.4.5.5
Yolov8OnnxRuntimeCPPInference: /usr/local/lib/libopencv_highgui.so.4.5.5
Yolov8OnnxRuntimeCPPInference: /usr/local/lib/libopencv_ml.so.4.5.5
Yolov8OnnxRuntimeCPPInference: /usr/local/lib/libopencv_objdetect.so.4.5.5
Yolov8OnnxRuntimeCPPInference: /usr/local/lib/libopencv_photo.so.4.5.5
Yolov8OnnxRuntimeCPPInference: /usr/local/lib/libopencv_stitching.so.4.5.5
Yolov8OnnxRuntimeCPPInference: /usr/local/lib/libopencv_video.so.4.5.5
Yolov8OnnxRuntimeCPPInference: /usr/local/lib/libopencv_videoio.so.4.5.5
Yolov8OnnxRuntimeCPPInference: /home/zy/Downloads/onnxruntime-linux-x64-gpu-1.10.0/lib/libonnxruntime.so
Yolov8OnnxRuntimeCPPInference: /usr/local/cuda/lib64/libcudart_static.a
Yolov8OnnxRuntimeCPPInference: /usr/lib/x86_64-linux-gnu/librt.so
Yolov8OnnxRuntimeCPPInference: /usr/local/lib/libopencv_imgcodecs.so.4.5.5
Yolov8OnnxRuntimeCPPInference: /usr/local/lib/libopencv_dnn.so.4.5.5
Yolov8OnnxRuntimeCPPInference: /usr/local/lib/libopencv_calib3d.so.4.5.5
Yolov8OnnxRuntimeCPPInference: /usr/local/lib/libopencv_features2d.so.4.5.5
Yolov8OnnxRuntimeCPPInference: /usr/local/lib/libopencv_flann.so.4.5.5
Yolov8OnnxRuntimeCPPInference: /usr/local/lib/libopencv_imgproc.so.4.5.5
Yolov8OnnxRuntimeCPPInference: /usr/local/lib/libopencv_core.so.4.5.5
Yolov8OnnxRuntimeCPPInference: CMakeFiles/Yolov8OnnxRuntimeCPPInference.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zy/vision/ultralytics/examples/YOLOv8-ONNXRuntime-CPP/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable Yolov8OnnxRuntimeCPPInference"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Yolov8OnnxRuntimeCPPInference.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Yolov8OnnxRuntimeCPPInference.dir/build: Yolov8OnnxRuntimeCPPInference

.PHONY : CMakeFiles/Yolov8OnnxRuntimeCPPInference.dir/build

CMakeFiles/Yolov8OnnxRuntimeCPPInference.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Yolov8OnnxRuntimeCPPInference.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Yolov8OnnxRuntimeCPPInference.dir/clean

CMakeFiles/Yolov8OnnxRuntimeCPPInference.dir/depend:
	cd /home/zy/vision/ultralytics/examples/YOLOv8-ONNXRuntime-CPP/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zy/vision/ultralytics/examples/YOLOv8-ONNXRuntime-CPP /home/zy/vision/ultralytics/examples/YOLOv8-ONNXRuntime-CPP /home/zy/vision/ultralytics/examples/YOLOv8-ONNXRuntime-CPP/build /home/zy/vision/ultralytics/examples/YOLOv8-ONNXRuntime-CPP/build /home/zy/vision/ultralytics/examples/YOLOv8-ONNXRuntime-CPP/build/CMakeFiles/Yolov8OnnxRuntimeCPPInference.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Yolov8OnnxRuntimeCPPInference.dir/depend
