cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

set(CMAKE_CUDA_COMPILER /usr/local/cuda-13.0/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 80)

# project name
project(polynomial_expansion LANGUAGES CXX CUDA)

# set(CMAKE_CXX_STANDARD 17)
# set(CXX_STANDARD 17)
# set(CUDA_STANDARD 17)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set your target GPU architecture (adjust if needed)
set(CMAKE_CUDA_ARCHITECTURES 120)

enable_testing()

add_library(inference_server_cutlass
    # ---- layers (CUDA) ----
    inference_server_cutlass/src/layers/embedding.cu
    # inference_server_cutlass/src/layers/linear.cu
    inference_server_cutlass/src/layers/lstmcell.cu
    inference_server_cutlass/src/layers/Encoder.cu

    # ---- utils ----
    inference_server_cutlass/src/utils/utils.cpp
    # src/utils/Model.cpp
    # src/utils/load_weights.cpp
    # src/utils/load_weights_old.cu
)

target_include_directories(inference_server_cutlass
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/inference_server_cutlass/include
)

# Add GoogleTest (assumes you have it in third_party/googletest)
add_subdirectory(external/googletest)

# CUTLASS root directory (update this path!)
set(CUTLASS_DIR "external/cutlass")

# Include CUTLASS headers
include_directories(
  ${CUTLASS_DIR}/include
  ${CUTLASS_DIR}/tools/util/include
)

# Add your test suite
add_subdirectory(tests/inference_server_cutlass)
