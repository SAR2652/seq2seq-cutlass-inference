#pragma once
#include "embedding.h"

class Decoder
{
    private:
        
        Decoder(const nlohmann::json encoder_metadata,
            WeightsMetadata* wmd);

        ~Decoder();
}