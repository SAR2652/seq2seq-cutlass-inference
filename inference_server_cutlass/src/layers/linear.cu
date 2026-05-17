#include "layers/gemm_config.h"
#include "layers/linear.h"


// Linear layer methods
template <typename KernelType, typename BiasType>
Linear<KernelType, BiasType>::Linear(
    const nlohmann::json linear_md, const WeightsMetadata& metadata)
{
    // std::cout << linear_md["bias"]["shape"] << std::endl;
    bias = extract_details_and_load_parameters<BiasType>(
        linear_md, "bias", metadata
    );

    // std::cout << linear_md["kernel"]["shape"] << std::endl;
    kernel = extract_details_and_load_parameters<KernelType>(
        linear_md, "kernel", metadata
    );
}


template <typename KernelType, typename BiasType>
Linear<KernelType, BiasType>::~Linear()
{
    cudaFree(bias);
    cudaFree(kernel);
}


template <typename KernelType, typename BiasType>
void Linear<KernelType, BiasType>::forward(
    KernelType*    input,
    BiasType*      output,
    int            total_tokens,
    int            input_size,
    int            output_size,
    cudaStream_t   stream        // <-- added
)
{
    using Gemm = typename GemmConfigSm80<KernelType, BiasType>::Gemm;

    typename Gemm::Arguments args{
        {total_tokens, output_size, input_size},  // M, N, K
        {input,  input_size},                     // A
        {kernel, input_size},                     // B
        {bias,   0},                              // C (broadcast)
        {output, output_size},                    // D
        {1, 1}                                    // alpha, beta
    };

    Gemm gemm_op;

    size_t workspace_size = gemm_op.get_workspace_size(args);
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMallocAsync(&workspace, workspace_size, stream);
    }

    cutlass::Status status = gemm_op.initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "GEMM initialize failed: " << int(status) << "\n";
        if (workspace) cudaFreeAsync(workspace, stream);
        return;
    }

    // Launch GEMM on the provided stream
    status = gemm_op(stream);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "GEMM launch failed: " << int(status) << "\n";
    }

    if (workspace) cudaFreeAsync(workspace, stream);
}

template class Linear<int8_t, int32_t>;
