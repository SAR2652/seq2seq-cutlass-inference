#include "utils/utils.cuh"
#include "layers/layer.cuh"
#include "layers/embedding.h"

// Embedding methods
Embedding::Embedding(const std::vector<int> shape,
                     const std::string dtype,
                     const float scale,
                     const int offset,
                     const int size,
                     const WeightsMetadata& metadata)
{
    this->shape = shape;
    this->dtype = dtype;
    this->scale = scale;

    std::vector<char> buffer = metadata.get_data(dtype, offset, size);

    if (dtype == "bfloat16")
    {
        embedding = load_to_device<__nv_bfloat16>(buffer);
    }
    else if (dtype == "int8")
    {
        embedding = load_to_device<int8_t>(buffer);
    }
    else if (dtype == "float32")
    {
        embedding = load_to_device<float>(buffer);
    }
    else
    {
        throw std::runtime_error("Unsupported dtype: " + dtype);
    }
}


// __restrict__ is a guarantee that the pointer points to only one specific
// memory location during its lifetime
template <typename T>
__global__ void embedding_lookup_kernel_t(
    const T* __restrict__ embedding_table,
    const int* __restrict__ input_indices,
    T* __restrict__ output,
    int embedding_dim,
    int sequence_length,
    int batch_size
)
{
    int token_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_tokens = batch_size * sequence_length;

    if (token_id < total_tokens) {
        int index = input_indices[token_id];
        for (int i = 0; i < embedding_dim; ++i) {
            output[token_id * embedding_dim + i] =
                embedding_table[index * embedding_dim + i];
        }
    }
}

template <typename T>
void forward_impl(void* embedding_raw,
                  int* input_indices,
                  int B, int S,
                  void* output_raw,
                  int embedding_dim,
                  cudaStream_t stream)        // <-- added
{
    T* embedding = static_cast<T*>(embedding_raw);
    T* output    = static_cast<T*>(output_raw);

    int total_tokens = B * S;
    int threads = 256;
    int blocks  = (total_tokens + threads - 1) / threads;

    embedding_lookup_kernel_t<T><<<blocks, threads, 0, stream>>>(   // <-- stream
        embedding,
        input_indices,
        output,
        embedding_dim,
        S,
        B
    );
}


void Embedding::forward(int* input_indices,
                        int batch_size,
                        int sequence_length,
                        void* output,
                        // int8_t* quantized_embedding_int8,
                        cudaStream_t stream)   // <-- added
{
    int embedding_dim = shape[1];
    float new_embedding_scale = scale / 255.0f;
    int total_tokens = batch_size * sequence_length;
    auto [blocks, threads] = get_threads_and_blocks(total_tokens, 256);

    if (dtype == "float16")
    {
        forward_impl<__half>(
            embedding,
            input_indices,
            batch_size,
            sequence_length,
            output,
            embedding_dim,
            stream
        );

        // quantize_to_int8<<<blocks, threads, 0, stream>>>(   // <-- stream
        //     static_cast<__half*>(output),
        //     quantized_embedding_int8,
        //     total_tokens,
        //     new_embedding_scale
        // );
    }
    else if (dtype == "bfloat16")
    {
        forward_impl<__nv_bfloat16>(
            embedding,
            input_indices,
            batch_size,
            sequence_length,
            output,
            embedding_dim,
            stream
        );

        // quantize_to_int8<<<blocks, threads, 0, stream>>>(   // <-- stream
        //     static_cast<__nv_bfloat16*>(output),
        //     quantized_embedding_int8,
        //     total_tokens,
        //     new_embedding_scale
        // );
    }
    else
    {
        throw std::runtime_error("Unsupported dtype: " + dtype);
    }
}



Embedding::~Embedding()
{
    cudaFree(embedding);
}
