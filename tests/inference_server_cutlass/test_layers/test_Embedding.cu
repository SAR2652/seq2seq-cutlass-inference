#include <iostream>
#include <string>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <typeinfo> // Required for typeid
#include "utils/utils.h"
#include "utils/utils.cuh"
#include "layers/embedding.h"


int main()
{
    // -------------------------
    // Create a CUDA stream
    // -------------------------
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::string common_path = "/home/sarvesh26/ML/seq2seq-cutlass-inference/";
    common_path += "output/";
    const std::string json_path = common_path + "metadata.json";
    const std::string bin_path = common_path + "weights.bin";

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

    std::vector<int> h_input_indices = {1, 6, 3, 2, 5, 4};

    int* d_input_indices;
    cudaMallocAsync(&d_input_indices, total_tokens * sizeof(int), stream);

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
    int mul_factor = (embedding_dtype == "float32")  ? sizeof(float)
                   : (embedding_dtype == "float16")  ? sizeof(__half)
                   : sizeof(__nv_bfloat16);

    int total_embedding_size = total_tokens * embedding_shape[1];

    cudaMallocAsync(
        &embedding_output,
        total_embedding_size * mul_factor,
        stream
    );

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
    // Cleanup (async)
    // -------------------------
    cudaFreeAsync(d_input_indices, stream);
    cudaFreeAsync(embedding_output, stream);

    // Ensure all work is done
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    delete embedding;
    delete wmd;

    return 0;
}
