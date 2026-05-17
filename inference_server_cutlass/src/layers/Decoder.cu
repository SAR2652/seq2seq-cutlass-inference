#include "layers/Decoder.h"
#include "utils/utils.cuh"

Decoder::Decoder()
{
    auto embedding_wmd = encoder_metadata["embedding"]["embedding"];
    const std::vector<int> embedding_shape =
        embedding_wmd["shape"].get<std::vector<int>>();
    embedding_dtype = embedding_wmd["dtype"];
    const int embedding_offset = embedding_wmd["offset"];
    const int embedding_size   = embedding_wmd["size"];
    embedding_dim = embedding_shape[1];

    // const float scale_x = wmd->metadata["calibration"]["scale_x"];

    embedding = new Embedding(
        embedding_shape, embedding_dtype,
        embedding_wmd["scale"], embedding_offset,
        embedding_size, *wmd
    );
}