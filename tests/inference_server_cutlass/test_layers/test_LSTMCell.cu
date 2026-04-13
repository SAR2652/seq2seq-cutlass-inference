#include <iostream>
#include <string>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <typeinfo> // Required for typeid
#include "utils/utils.h"
#include "utils/utils.cuh"
#include "layers/embedding.h"
#include "layers/lstmcell.h"





int main()
{
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

    // scale_x is the calibrated activation scale for the embedding output.
    // It was derived by running real data through the embedding layer during
    // calibration and is the correct scale for both the BF16→INT8 quantization
    // step and the x_quant_scale argument to LSTMCell::forward().
    // embedding_wmd["scale"] is the weight-tensor range (max-min) and must
    // NOT be used here.
    const float scale_x = wmd->metadata["calibration"]["scale_x"];

    Embedding* embedding = new Embedding(
        embedding_shape, embedding_dtype,
        embedding_wmd["scale"], embedding_offset,
        embedding_size, *wmd
    );

    // -------------------------
    // Prepare input
    // -------------------------
    const int batch_size = 6;
    const int sequence_length = 1;
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

    // Allocate embedding output
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
    // Allocate quantized int8 buffer
    // -------------------------
    int8_t* quantized_embedding_int8;
    cudaMallocAsync(
        &quantized_embedding_int8,
        total_embedding_size * sizeof(int8_t),
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
        stream
    );

    // -------------------------
    // Quantize embedding output → INT8 using scale_x
    // -------------------------
    {
        int block = 256;
        int grid  = (total_embedding_size + block - 1) / block;
        float inv_scale = 1.0f / scale_x;
        if (embedding_dtype == "bfloat16") {
            quantize_to_int8<<<grid, block, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(embedding_output),
                quantized_embedding_int8,
                total_embedding_size,
                inv_scale
            );
        } else if (embedding_dtype == "float16") {
            quantize_to_int8<<<grid, block, 0, stream>>>(
                static_cast<const __half*>(embedding_output),
                quantized_embedding_int8,
                total_embedding_size,
                inv_scale
            );
        } else if (embedding_dtype == "float32") {
            quantize_to_int8<<<grid, block, 0, stream>>>(
                static_cast<const float*>(embedding_output),
                quantized_embedding_int8,
                total_embedding_size,
                inv_scale
            );
        } else {
            throw std::runtime_error("Unsupported embedding dtype: " + embedding_dtype);
        }
    }

    // -------------------------
    // LSTMCell
    // -------------------------
    auto encoder_fwd_lstm_wmd = wmd->metadata["encoder"]["forward_lstm"];

    auto kernel_tag = dtype_to_tag(
        encoder_fwd_lstm_wmd["hf"]["kernel"]["dtype"]
    );
    auto bias_tag = dtype_to_tag(
        encoder_fwd_lstm_wmd["hf"]["bias"]["dtype"]
    );

    if (kernel_tag == DTypeTag::Int8 && bias_tag == DTypeTag::Int32)
    {
        using KernelType = int8_t;
        using BiasType   = int32_t;

        auto* lstmcell = new LSTMCell<KernelType, BiasType>(
            encoder_fwd_lstm_wmd, *wmd
        );

        // hidden_dim is derived from kernel shape inside LSTMCell; read it
        // from metadata directly here for allocation
        // (column-major: shape[0] = hidden_dim)
        const int hidden_dim =
            encoder_fwd_lstm_wmd["if"]["kernel"]["shape"][0].get<int>();

        // h and c stay float32 for the full sequence; never quantized between
        // steps
        float* fwd_hidden;
        float* fwd_cell;
        float* new_hidden;
        float* new_cell;

        cudaMallocAsync(&fwd_hidden, batch_size * hidden_dim * sizeof(float),
                        stream);
        cudaMallocAsync(&fwd_cell,   batch_size * hidden_dim * sizeof(float),
                        stream);
        cudaMallocAsync(&new_hidden, batch_size * hidden_dim * sizeof(float),
                        stream);
        cudaMallocAsync(&new_cell,   batch_size * hidden_dim * sizeof(float),
                        stream);

        // Zero-initialise: initial h and c are zero at the start of each
        // sequence
        cudaMemsetAsync(fwd_hidden, 0, batch_size * hidden_dim * sizeof(float),
                        stream);
        cudaMemsetAsync(fwd_cell,   0, batch_size * hidden_dim * sizeof(float),
                        stream);

        lstmcell->forward(
            fwd_cell,
            fwd_hidden,
            quantized_embedding_int8,
            new_cell,
            new_hidden,
            batch_size,
            scale_x,
            stream
        );

        delete lstmcell;

        cudaFreeAsync(fwd_hidden, stream);
        cudaFreeAsync(fwd_cell,   stream);
        cudaFreeAsync(new_hidden, stream);
        cudaFreeAsync(new_cell,   stream);
    }
    else
    {
        throw std::runtime_error("Unsupported kernel/bias dtype combination");
    }

    // -------------------------
    // Cleanup
    // -------------------------
    cudaFreeAsync(d_input_indices, stream);
    cudaFreeAsync(embedding_output, stream);
    cudaFreeAsync(quantized_embedding_int8, stream);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    delete embedding;
    delete wmd;

    return 0;
}
