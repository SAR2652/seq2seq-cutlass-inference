#pragma once
#include "../utils/utils.h"
#include <cuda_runtime.h>


class Layer {
    protected:
        template <typename T>
        T* load_to_device(const std::vector<char>& buffer) const;

        template <typename T>
        T* extract_details_and_load_parameters(nlohmann::json param_md,
            const std::string& param_type, const WeightsMetadata& wmd) const;

    public:
        Layer() = default;
        virtual ~Layer() = default;
};
