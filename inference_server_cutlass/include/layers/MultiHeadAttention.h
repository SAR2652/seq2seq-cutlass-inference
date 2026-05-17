template<typename KernelType, typename BiasType>
class MultiHeadAttention
{
    private:
        Linear<KernelType, BiasType> *query_proj, *key_proj, *value_proj;
        Linear<KernelType, BiasType> *out_proj;

    public:
        MultiHeadAttention(const nlohmann::json mha_metadata,
               const WeightsMetadata& metadata);

        ~MultiHeadAttention();

        // void forward(KernelType* input, BiasType* output,
        //     const int total_tokens, const int input_shape,
        //     const int output_shape, cudaStream_t stream);

};