#include <iostream>
#include <string>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <typeinfo>
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
    const int embedding_dim    = embedding_shape[1];

    const float scale_x = wmd->metadata["calibration"]["scale_x"];

    Embedding* embedding = new Embedding(
        embedding_shape, embedding_dtype,
        embedding_wmd["scale"], embedding_offset,
        embedding_size, *wmd
    );

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

    // Allocate embedding output  [total_tokens, embedding_dim]
    int mul_factor = (embedding_dtype == "float32")  ? sizeof(float)
                   : (embedding_dtype == "float16")  ? sizeof(__half)
                   : sizeof(__nv_bfloat16);

    const int total_embedding_size = total_tokens * embedding_dim;

    void* embedding_output = nullptr;
    cudaMallocAsync(&embedding_output, total_embedding_size * mul_factor, stream);

    // Quantized int8 buffer  [total_tokens, embedding_dim]
    int8_t* quantized_embedding_int8;
    cudaMallocAsync(
        &quantized_embedding_int8,
        total_embedding_size * sizeof(int8_t),
        stream
    );

    // -------------------------
    // Run embedding forward for all tokens at once
    // -------------------------
    embedding->forward(
        d_input_indices,
        batch_size,
        seq_len,
        embedding_output,
        stream
    );

    // -------------------------
    // Quantize entire embedding output → INT8
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
    // LSTMCell sequence loop
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

        const int hidden_dim =
            encoder_fwd_lstm_wmd["if"]["kernel"]["shape"][0].get<int>();

        // Double-buffer h and c: fwd_* is the current state,
        // new_* receives the next state; pointers are swapped after each step.
        float* fwd_hidden;
        float* fwd_cell;
        float* new_hidden;
        float* new_cell;

        cudaMallocAsync(&fwd_hidden, batch_size * hidden_dim * sizeof(float), stream);
        cudaMallocAsync(&fwd_cell,   batch_size * hidden_dim * sizeof(float), stream);
        cudaMallocAsync(&new_hidden, batch_size * hidden_dim * sizeof(float), stream);
        cudaMallocAsync(&new_cell,   batch_size * hidden_dim * sizeof(float), stream);

        // Zero-initialise h_0 and c_0
        cudaMemsetAsync(fwd_hidden, 0, batch_size * hidden_dim * sizeof(float), stream);
        cudaMemsetAsync(fwd_cell,   0, batch_size * hidden_dim * sizeof(float), stream);

        // outputs[t] holds fwd_hidden after step t  →  shape [seq_len, batch, hidden]
        float* outputs;
        cudaMallocAsync(
            &outputs,
            seq_len * batch_size * hidden_dim * sizeof(float),
            stream
        );

        // -------------------------
        // Iterate over sequence  (mirrors the Python loop above)
        // -------------------------
        const int step_embedding_elems = batch_size * embedding_dim;

        for (int t = 0; t < seq_len; ++t) {
            // x_t: quantized embeddings for this timestep [batch, embedding_dim]
            const int8_t* x_t =
                quantized_embedding_int8 + t * step_embedding_elems;

            lstmcell->forward(
                fwd_cell,
                fwd_hidden,
                x_t,
                new_cell,
                new_hidden,
                batch_size,
                scale_x,
                stream
            );

            // outputs.append(fwd_hidden)  →  copy new_hidden into outputs[t]
            cudaMemcpyAsync(
                outputs + t * batch_size * hidden_dim,
                new_hidden,
                batch_size * hidden_dim * sizeof(float),
                cudaMemcpyDeviceToDevice,
                stream
            );

            // Advance state: (fwd_hidden, fwd_cell) ← (new_hidden, new_cell)
            std::swap(fwd_hidden, new_hidden);
            std::swap(fwd_cell,   new_cell);
        }

        // After the loop fwd_hidden / fwd_cell hold the final hidden state.

        cudaStreamSynchronize(stream);

        // -------------------------
        // Optional: print final hidden state (first batch item, first 8 values)
        // -------------------------
        {
            std::vector<float> h_final(batch_size * hidden_dim);
            cudaMemcpy(
                h_final.data(),
                fwd_hidden,
                batch_size * hidden_dim * sizeof(float),
                cudaMemcpyDeviceToHost
            );
            std::cout << "Final hidden state (batch 0, first 8 dims):\n  ";
            for (int i = 0; i < std::min(8, hidden_dim); ++i)
                std::cout << h_final[i] << " ";
            std::cout << "\n";
        }

        delete lstmcell;

        cudaFreeAsync(fwd_hidden, stream);
        cudaFreeAsync(fwd_cell,   stream);
        cudaFreeAsync(new_hidden, stream);
        cudaFreeAsync(new_cell,   stream);
        cudaFreeAsync(outputs,    stream);
    }
    else
    {
        throw std::runtime_error("Unsupported kernel/bias dtype combination");
    }

    // -------------------------
    // Cleanup
    // -------------------------
    cudaFreeAsync(d_input_indices,         stream);
    cudaFreeAsync(embedding_output,        stream);
    cudaFreeAsync(quantized_embedding_int8, stream);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    delete embedding;
    delete wmd;

    return 0;
}
