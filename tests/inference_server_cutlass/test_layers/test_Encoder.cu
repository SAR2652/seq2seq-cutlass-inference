#include <iostream>
#include <string>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <typeinfo>
#include "utils/utils.h"
#include "utils/utils.cuh"
#include "layers/Encoder.h"



int main()
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::string common_path = "/home/sar26/ML/Polynomial-Expansion/output/";
    std::string json_path = common_path + "metadata.json";
    std::string bin_path = common_path + "weights.bin";

    WeightsMetadata* wmd = new WeightsMetadata(json_path, bin_path);

    // -------------------------
    // Prepare input
    // Input indices are laid out [seq, batch] so that at timestep t the
    // contiguous slice [t*batch_size .. (t+1)*batch_size) holds all batch
    // items for that step — matching embeddings[:, t, :] in PyTorch.
    // -------------------------
    const int batch_size = 6;
    const int seq_len    = 4;
    const int total_tokens = batch_size * seq_len;   // 24

    // Layout: row 0 = t=0 tokens, row 1 = t=1 tokens, …
    std::vector<int> h_input_indices = {
        1, 6, 3, 2, 5, 4,   // t=0
        2, 5, 1, 4, 3, 6,   // t=1
        3, 4, 2, 6, 1, 5,   // t=2
        4, 3, 6, 1, 2, 5,   // t=3
    };

    int* d_input_indices;
    cudaMallocAsync(&d_input_indices, total_tokens * sizeof(int), stream);
    cudaMemcpyAsync(
        d_input_indices,
        h_input_indices.data(),
        total_tokens * sizeof(int),
        cudaMemcpyHostToDevice,
        stream
    );

    //--------------------------
    // Initialize and run Encoder
    //--------------------------
    auto encoder_wmd = wmd->metadata["encoder"];
    Encoder* encoder = new Encoder(encoder_wmd, wmd);
    const float scale_x = wmd->metadata["calibration"]["scale_x"];

    float* encoder_outputs;
    cudaMallocAsync(
        &encoder_outputs,
        seq_len * batch_size * encoder->output_hidden_dim() * sizeof(float),
        stream
    );

    encoder->forward(d_input_indices, encoder_outputs, batch_size, seq_len,
        scale_x, stream);

    // -------------------------
    // Cleanup
    // -------------------------
    delete encoder;
    cudaFreeAsync(d_input_indices, stream);
    cudaFreeAsync(encoder_outputs, stream);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    return 0;
}
