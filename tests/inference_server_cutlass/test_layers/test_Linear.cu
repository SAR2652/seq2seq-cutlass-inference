#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <typeinfo> // Required for typeid
#include "utils/utils.h"
#include "utils/utils.cuh"
#include "layers/embedding.h"
#include "layers/linear.h"



int main()
{
    // -------------------------
    // Create a CUDA stream
    // -------------------------
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::string common_path = "/home/sar26/ML/Polynomial-Expansion/output/";
    std::string json_path = common_path + "metadata.json";
    std::string bin_path = common_path + "weights.bin";

    WeightsMetadata* wmd = new WeightsMetadata(json_path, bin_path);

    auto embedding_wmd = wmd->metadata["encoder"]["embedding"]["embedding"];
    const std::vector<int> embedding_shape =
        embedding_wmd["shape"].get<std::vector<int>>();
    const std::string embedding_dtype = embedding_wmd["dtype"];
    const int embedding_offset = embedding_wmd["offset"];
    const int embedding_size   = embedding_wmd["size"];
    const float embedding_scale = embedding_wmd["scale"];

    Embedding* embedding = new Embedding(
        embedding_shape, embedding_dtype,
        embedding_scale, embedding_offset,
        embedding_size, *wmd
    );

    // -------------------------
    // Prepare input
    // -------------------------
    const int batch_size = 2;
    const int sequence_length = 3;
    int total_tokens = batch_size * sequence_length;

    std::vector<int> h_input_indices = {0, 1, 2, 1, 0, 3};

    int* d_input_indices;
    
    // cudaMallocAsync / cudaFreeAsync ensure memory ops don’t force
    // device‑wide synchronization.
    cudaMallocAsync(&d_input_indices, total_tokens * sizeof(int), stream);

    
    // cudaMemcpyAsync lets transfers overlap with compute.
    cudaMemcpyAsync(
        d_input_indices,
        h_input_indices.data(),
        total_tokens * sizeof(int),
        cudaMemcpyHostToDevice,
        stream
    );

    // -------------------------
    // Allocate embedding output
    // -------------------------
    void* embedding_output = nullptr;
    int mul_factor = (embedding_dtype == "float16")
                        ? sizeof(__half)
                        : sizeof(__nv_bfloat16);

    int total_embedding_size = total_tokens * embedding_shape[1];

    cudaMallocAsync(
        &embedding_output,
        total_embedding_size * mul_factor,
        stream
    );

    // -------------------------
    // Allocate quantized int8 buffer
    // -------------------------
    // int8_t* quantized_embedding_int8;
    // cudaMallocAsync(
    //     &quantized_embedding_int8,
    //     total_embedding_size * sizeof(int8_t),
    //     stream
    // );

    // -------------------------
    // Run embedding forward (async)
    // -------------------------
    embedding->forward(
        d_input_indices,
        batch_size,
        sequence_length,
        embedding_output,
        // quantized_embedding_int8,
        stream
    );

    // -------------------------
    // Linear layer
    // -------------------------
    auto linear_wmd = wmd->metadata["decoder"]["fc_out"];

    auto kernel_tag = dtype_to_tag(linear_wmd["kernel"]["dtype"]);
    auto bias_tag   = dtype_to_tag(linear_wmd["bias"]["dtype"]);

    int linear_shape = linear_wmd["bias"]["shape"][0];
    int linear_output_size = total_tokens * linear_shape;

    void* linear_output = nullptr;

    if (kernel_tag == DTypeTag::Int8 && bias_tag == DTypeTag::Int32)
    {
        using KernelType = int8_t;
        using BiasType   = int32_t;

        cudaMallocAsync(
            &linear_output,
            linear_output_size * sizeof(BiasType),
            stream
        );

        auto* linear = new Linear<KernelType, BiasType>(
            linear_wmd, *wmd, stream
        );

        linear->forward(
            quantized_embedding_int8,
            static_cast<BiasType*>(linear_output),
            total_tokens,
            embedding_shape[1],
            linear_shape,
            stream
        );

        delete linear;
    }
    else {
        throw std::runtime_error("Unsupported kernel/bias dtype combination");
    }

    // -------------------------
    // Cleanup (async)
    // -------------------------
    cudaFreeAsync(d_input_indices, stream);
    cudaFreeAsync(embedding_output, stream);
    cudaFreeAsync(quantized_embedding_int8, stream);
    cudaFreeAsync(linear_output, stream);

    // Ensure all work is done
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    delete embedding;
    delete wmd;

    return 0;
}
