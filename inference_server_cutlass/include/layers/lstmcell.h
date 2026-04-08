#include "layer.h"


template<typename KernelType, typename BiasType>
class LSTMCell: public Layer
{
    private:
        // Hidden state Kernel and Bias
        KernelType *hf_kernel, *hg_kernel, *hi_kernel, *ho_kernel;
        BiasType *hf_bias, *hg_bias, *hi_bias, *ho_bias;

        // Input Kernels only
        KernelType *if_kernel, *ig_kernel, *ii_kernel, *io_kernel;

        // Bias scale values
        float hf_bias_scale, hg_bias_scale, hi_bias_scale, ho_bias_scale;

        // Hidden kernel scale values
        float hf_kernel_scale, hg_kernel_scale, hi_kernel_scale, ho_kernel_scale;

        // Input kernel scale values
        float if_kernel_scale, ig_kernel_scale, ii_kernel_scale, io_kernel_scale;

        // Dimensions (derived from kernel shapes in metadata)
        int hidden_dim, input_dim;

        // Calibrated scale for transiently quantizing h (float32 → int8) each step.
        // h itself is always stored as float32; this scale is only used at GEMM boundaries.
        float h_quant_scale;

        // Four persistent streams — one per gate (i, f, g, o).
        // Pre-created to avoid per-call overhead; gate computations run in parallel.
        cudaStream_t gate_streams_[4];

    public:
        LSTMCell(const nlohmann::json lstm_metadata,
            const WeightsMetadata& metadata);

        ~LSTMCell();

        // x_quant_scale: quantization scale of the int8 input embeddings (from the
        // embedding layer). Used to compute the effective dequantization scale for x@Wx.
        void forward(float* c, float* h, const KernelType* x, float* new_c,
                    float* new_h, int batch_size, float x_quant_scale,
                    cudaStream_t stream);
};
