cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(generate_depth LANGUAGES CXX CUDA)

# if you use conda env,gedit it as your conda env 
set(CMAKE_PREFIX_PATH /home/wxb/miniconda3/envs/wxb)
set(PYTHON_LIBRARIES /home/wxb/miniconda3/envs/wxb/lib)
set(PYTHON_INCLUDE_DIRS /home/wxb/miniconda3/envs/wxb/include/python3.8)

find_package(Python REQUIRED)
find_package(CUDA REQUIRED)

execute_process(
    COMMAND
        ${Python_EXECUTABLE} -c
            "import torch.utils; print(torch.utils.cmake_prefix_path)"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE DCMAKE_PREFIX_PATH)

set(CMAKE_PREFIX_PATH ${DCMAKE_PREFIX_PATH})
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS ${TORCH_INSTALL_PREFIX}/lib)
# modify to your own python path, or delete it if ok
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(SRCS generate_depth_pts.cpp generate_depth_pts.cu)
add_library(generate_depth SHARED ${SRCS})

target_link_libraries(generate_depth "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIBRARY}")