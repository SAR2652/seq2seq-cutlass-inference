#include "layers/linear.h"

template <typename KernelType, typename BiasType>
MultiheadAttention::MultiHeadAttention(const nlohmann::json mha_metadata,
               const WeightsMetadata& metadata)
{
    query_proj_wmd = mha_metadata["query_proj"];
    query_proj = new Linear<KernelType, BiasType>(
        query_proj_wmd, metadata
    );

    key_proj_wmd = mha_metadata["key_proj"];
    key_proj = new Linear<KernelType, BiasType>(
        key_proj_wmd, metadata
    );

    value_proj_wmd = mha_metadata["value_proj"];
    value_proj = new Linear<KernelType, BiasType>(
        value_proj_wmd, metadata
    );

    out_proj = mha_metadata["out_proj"];
    out_proj = new Linear<KernelType, BiasType>(
        out_proj_wmd, metadata
    );

}