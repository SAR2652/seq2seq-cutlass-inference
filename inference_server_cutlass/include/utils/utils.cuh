#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <stdio.h>

#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) \
    { \
        printf("CUDA Error: %s (at %s:%d)\n", cudaGetErrorString(err), \
        __FILE__, __LINE__); \
        return -1; \
    } \
}

#endif

std::tuple<int, int> get_threads_and_blocks(int total_tokens,
                                            int threads = 256);

template <typename T>
__device__ inline float to_float(T x) {
    return static_cast<float>(x);
}

template <>
__device__ inline float to_float<__half>(__half x) {
    return __half2float(x);
}

template <>
__device__ inline float to_float<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

// Works for float, __half, and __nv_bfloat16.
// inv_scale = 1.0f / scale (precomputed by caller to avoid per-element division).
template <typename T>
__global__ void quantize_to_int8(const T* input, int8_t* output,
                                  int N, float inv_scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float v = rintf(to_float(input[idx]) * inv_scale);
        output[idx] = static_cast<int8_t>(fmaxf(-128.f, fminf(127.f, v)));
    }
}
