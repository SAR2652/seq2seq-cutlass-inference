#include "layer.h"


template <typename T>
T* Layer::load_to_device(const std::vector<char>& buffer) const
{
    size_t num_elements = buffer.size() / sizeof(T);
    T* device_ptr = nullptr;

    cudaError_t err = cudaMalloc(&device_ptr,
        num_elements * sizeof(T)
    );
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMalloc failed: "
        + std::string(cudaGetErrorString(err)));
    }

    err = cudaMemcpy(device_ptr, buffer.data(),
        num_elements * sizeof(T), cudaMemcpyHostToDevice
    );
    if (err != cudaSuccess) {
        cudaFree(device_ptr);
        throw std::runtime_error("cudaMemcpy failed: " +
        std::string(cudaGetErrorString(err)));
    }

    return device_ptr;
}

template <typename T>
T* Layer::extract_details_and_load_parameters(
    nlohmann::json param_md, const std::string& param_type,
    const WeightsMetadata& wmd) const
{
    int offset = param_md[param_type]["offset"];
    int size = param_md[param_type]["size"];
    std::string dtype = param_md[param_type]["dtype"];
    std::vector<char> buffer = wmd.get_data(dtype, offset, size);
    T* device_ptr = load_to_device<T>(buffer);
    return device_ptr;
}
