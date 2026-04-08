#include "layers/layer.cuh"
#include "layers/lstmcell.h"
#include "layers/gemm_config.h"


// LSTMCell methods
template <typename KernelType, typename BiasType>
LSTMCell<KernelType, BiasType>::LSTMCell(
    const nlohmann::json lstm_metadata, const WeightsMetadata& metadata)
{
    // Dimensions from if_kernel shape (column-major: [hidden_dim, input_dim])
    hidden_dim = lstm_metadata["if"]["kernel"]["shape"][0];
    input_dim  = lstm_metadata["if"]["kernel"]["shape"][1];

    // Calibrated scale for transiently quantizing h (float32 → int8) before GEMMs.
    // h is always stored as float32; this scale is only applied at GEMM boundaries.
    h_quant_scale = lstm_metadata["h_scale"];

    // HF
    auto hf_md = lstm_metadata["hf"];
    hf_bias = extract_details_and_load_parameters<BiasType>(
        hf_md, "bias", metadata
    );
    hf_bias_scale   = lstm_metadata["hf"]["bias"]["scale"];
    hf_kernel_scale = lstm_metadata["hf"]["kernel"]["scale"];
    hf_kernel = extract_details_and_load_parameters<KernelType>(
        hf_md, "kernel", metadata
    );

    // HG
    auto hg_md = lstm_metadata["hg"];
    hg_bias = extract_details_and_load_parameters<BiasType>(
        hg_md, "bias", metadata
    );
    hg_bias_scale   = lstm_metadata["hg"]["bias"]["scale"];
    hg_kernel_scale = lstm_metadata["hg"]["kernel"]["scale"];
    hg_kernel = extract_details_and_load_parameters<KernelType>(
        hg_md, "kernel", metadata
    );

    // HI
    auto hi_md = lstm_metadata["hi"];

    hi_bias = extract_details_and_load_parameters<BiasType>(
        hi_md, "bias", metadata
    );
    hi_bias_scale   = lstm_metadata["hi"]["bias"]["scale"];
    hi_kernel_scale = lstm_metadata["hi"]["kernel"]["scale"];
    hi_kernel = extract_details_and_load_parameters<KernelType>(
        hi_md, "kernel", metadata
    );

    // HO
    auto ho_md = lstm_metadata["ho"];

    ho_bias = extract_details_and_load_parameters<BiasType>(
        ho_md, "bias", metadata
    );
    ho_bias_scale   = lstm_metadata["ho"]["bias"]["scale"];
    ho_kernel_scale = lstm_metadata["ho"]["kernel"]["scale"];
    ho_kernel = extract_details_and_load_parameters<KernelType>(
        ho_md, "kernel", metadata
    );

    // IF
    auto if_md = lstm_metadata["if"];
    if_kernel_scale = lstm_metadata["if"]["kernel"]["scale"];
    if_kernel = extract_details_and_load_parameters<KernelType>(
        if_md, "kernel", metadata
    );

    // IG
    auto ig_md = lstm_metadata["ig"];
    ig_kernel_scale = lstm_metadata["ig"]["kernel"]["scale"];
    ig_kernel = extract_details_and_load_parameters<KernelType>(
        ig_md, "kernel", metadata
    );

    // II
    auto ii_md = lstm_metadata["ii"];
    ii_kernel_scale = lstm_metadata["ii"]["kernel"]["scale"];
    ii_kernel = extract_details_and_load_parameters<KernelType>(
        ii_md, "kernel", metadata
    );

    // IO
    auto io_md = lstm_metadata["io"];
    io_kernel_scale = lstm_metadata["io"]["kernel"]["scale"];
    io_kernel = extract_details_and_load_parameters<KernelType>(
        io_md, "kernel", metadata
    );

    // One non-blocking stream per gate so all four gates can run in parallel.
    // cudaStreamNonBlocking ensures they don't implicitly synchronize with the
    // legacy (NULL) stream, keeping them fully independent from each other and
    // from the caller's stream.
    for (int i = 0; i < 4; i++)
        cudaStreamCreateWithFlags(&gate_streams_[i], cudaStreamNonBlocking);
}

template <typename KernelType, typename BiasType>
LSTMCell<KernelType, BiasType>::~LSTMCell()
{
    cudaFreeAsync(hf_bias, stream_);
    cudaFreeAsync(hf_kernel, stream_);

    cudaFreeAsync(hg_bias, stream_);
    cudaFreeAsync(hg_kernel, stream_);

    cudaFreeAsync(hi_bias, stream_);
    cudaFreeAsync(hi_kernel, stream_);

    cudaFreeAsync(ho_bias, stream_);
    cudaFreeAsync(ho_kernel, stream_);

    cudaFreeAsync(if_kernel, stream_);
    cudaFreeAsync(ig_kernel, stream_);
    cudaFreeAsync(ii_kernel, stream_);
    cudaFreeAsync(io_kernel, stream_);

    for (int i = 0; i < 4; i++)
        cudaStreamDestroy(gate_streams_[i]);
}


// Broadcast a 1-D bias row [N] into a 2-D matrix [M, N].
// Each thread writes one element: out[row * N + col] = bias[col].
template <typename BiasType>
__global__ void broadcast_bias_kernel(
    BiasType       *out,    // [M, N] destination
    const BiasType *bias,   // [N]    source (one row)
    int             M,
    int             N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    out[idx] = bias[idx % N];
}


// Quantize a float32 vector to int8 using a fixed scale.
// int8_val = clamp(round(float_val / scale), -128, 127)
// inv_scale = 1.0f / scale is precomputed by the caller to avoid repeated division.
__global__ void quantize_fp32_to_int8_kernel(
    int8_t      *out,
    const float *in,
    float        inv_scale,
    int          total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    float v = rintf(in[idx] * inv_scale);
    out[idx] = (int8_t)fmaxf(-128.f, fminf(127.f, v));
}


// Dequantize two int32 GEMM results with their own independent scales,
// sum them in float32, then apply sigmoid or tanh.
//   gate = act( xWx_int32 * scale_xWx  +  hWh_b_int32 * scale_hWh_b )
__global__ void add_and_act_kernel(
    float          *gate_out,   // [B, hidden_dim] float output
    const int32_t  *xWx,        // [B, hidden_dim] int32: result of x @ Wx
    const int32_t  *hWh_b,      // [B, hidden_dim] int32: result of h_int8 @ Wh + b
    int             total,
    float           scale_xWx,  // x_quant_scale * Wx_kernel_scale
    float           scale_hWh_b,// h_quant_scale * Wh_kernel_scale
    bool            use_sigmoid)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    float v = (float)xWx[idx] * scale_xWx + (float)hWh_b[idx] * scale_hWh_b;
    if (use_sigmoid) v = 1.f / (1.f + expf(-v));
    else             v = tanhf(v);
    gate_out[idx] = v;
}


// Compute one LSTM gate in int8 arithmetic with correct per-GEMM dequantization.
//
// The gate value (pre-activation) is:
//   gate_float = (x @ Wx)_int32 * scale_xWx  +  (h_int8 @ Wh + b)_int32 * scale_hWh_b
//
// Two separate scales are required because the two GEMMs accumulate on different
// numeric ranges: scale_xWx = x_quant_scale * Wx_kernel_scale, and
// scale_hWh_b = h_quant_scale * Wh_kernel_scale.  Adding the raw int32 results
// and applying a single scale (the previous approach) is only valid when both
// effective scales are equal, which calibration cannot guarantee.
//
// h is kept as float32 between timesteps.  It is quantized to int8 transiently
// here, used for the GEMM, and the int8 buffer is immediately freed.
template <typename KernelType, typename BiasType>
void run_gate(
    const KernelType *x,           // [B, input_dim]   int8
    const float      *h,           // [B, hidden_dim]  float32 (stored state)
    const KernelType *Wx,          // [input_dim, hidden_dim]   int8
    const KernelType *Wh,          // [hidden_dim, hidden_dim]  int8
    const BiasType   *bh,          // [hidden_dim]  int32
    float            *gate_out,    // [B, hidden_dim]  float32 output
    int  batch_size,
    int  input_dim,
    int  hidden_dim,
    bool use_sigmoid,
    float scale_xWx,               // x_quant_scale * Wx_kernel_scale
    float scale_hWh_b,             // h_quant_scale * Wh_kernel_scale
    float h_quant_scale,           // scale used to quantize h → int8
    cudaStream_t stream)
{
    using Gemm = typename GemmConfigSm80<KernelType, BiasType>::Gemm;

    int M  = batch_size;
    int N  = hidden_dim;
    int Kx = input_dim;
    int Kh = hidden_dim;

    BiasType   *xWx_buf, *hWh_b_buf;
    KernelType *h_int8;
    cudaMallocAsync(&xWx_buf,   sizeof(BiasType)   * M * N,  stream);
    cudaMallocAsync(&hWh_b_buf, sizeof(BiasType)   * M * N,  stream);
    cudaMallocAsync(&h_int8,    sizeof(KernelType) * M * Kh, stream);

    // 1) Quantize h (float32) → h_int8 transiently.
    //    h itself stays float32; this copy is only alive for the duration of this call.
    {
        int total  = M * Kh;
        int block  = 256;
        int grid   = (total + block - 1) / block;
        float inv_scale = 1.0f / h_quant_scale;
        quantize_fp32_to_int8_kernel<<<grid, block, 0, stream>>>(
            h_int8, h, inv_scale, total);
    }

    // 2) xWx_buf = x @ Wx  (int8 × int8 → int32, no bias)
    {
        Gemm gemm_op;
        typename Gemm::Arguments args(
            {M, N, Kx},
            {x,  Kx},
            {Wx, N},
            {nullptr, N},
            {xWx_buf, N},
            {1.0f, 0.0f}
        );
        gemm_op.initialize(args, nullptr, stream);
        gemm_op(stream);
    }

    // 3) hWh_b_buf = h_int8 @ Wh + bh
    //    Broadcast bias row [N] across all M rows of hWh_b_buf [M, N],
    //    then fuse into GEMM with beta=1.
    //    cudaMemcpy2DAsync cannot be used here: its src_pitch > 0 means it would
    //    read M rows from bh, but bh has only one row (N elements) — a buffer overread.
    {
        int total = M * N;
        int block = 256;
        int grid  = (total + block - 1) / block;
        broadcast_bias_kernel<BiasType><<<grid, block, 0, stream>>>(
            hWh_b_buf, bh, M, N);
    }
    {
        Gemm gemm_op;
        typename Gemm::Arguments args(
            {M, N, Kh},
            {h_int8, Kh},
            {Wh, N},
            {hWh_b_buf, N},   // C = broadcast bias
            {hWh_b_buf, N},   // D = h_int8 @ Wh + bias
            {1.0f, 1.0f}
        );
        gemm_op.initialize(args, nullptr, stream);
        gemm_op(stream);
    }

    // 4) Dequantize each GEMM result with its own scale, sum, activate.
    int total = M * N;
    int block = 256;
    int grid  = (total + block - 1) / block;
    add_and_act_kernel<<<grid, block, 0, stream>>>(
        gate_out, xWx_buf, hWh_b_buf, total, scale_xWx, scale_hWh_b, use_sigmoid);

    cudaFreeAsync(xWx_buf,   stream);
    cudaFreeAsync(hWh_b_buf, stream);
    cudaFreeAsync(h_int8,    stream);
}


__global__ void lstm_update_kernel(
    const float *c,
    const float *i_gate,
    const float *f_gate,
    const float *g_gate,
    const float *o_gate,
    float *new_c,
    float *new_h,
    int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    float ci = c[idx];
    float i  = i_gate[idx];
    float f  = f_gate[idx];
    float g  = g_gate[idx];
    float o  = o_gate[idx];

    float c_prime = f * ci + i * g;
    float h_prime = o * tanhf(c_prime);

    new_c[idx] = c_prime;
    new_h[idx] = h_prime;
}


template <typename KernelType, typename BiasType>
void LSTMCell<KernelType, BiasType>::forward(
    float            *c,            // [B, hidden_dim] cell state (in/out via new_c)
    float            *h,            // [B, hidden_dim] hidden state (in/out via new_h)
    const KernelType *x,            // [B, input_dim]  int8 input embeddings
    float            *new_c,        // [B, hidden_dim] updated cell state (out)
    float            *new_h,        // [B, hidden_dim] updated hidden state (out)
    int               batch_size,
    float             x_quant_scale,// quantization scale of the int8 input x
    cudaStream_t      stream)
{
    int B = batch_size;
    int H = hidden_dim;

    // Allocate gate output buffers on the caller's stream.
    // A "setup done" event lets the four gate streams wait for these
    // allocations before they start writing into the buffers.

    // Alternative: allocate each buffer on its respective gate_streams_[i].
    // That would eliminate the setup_done event below, because an alloc and
    // its first write would be on the same stream (ordered automatically).
    // cudaFreeAsync can still happen on a different stream (the caller's),
    // as long as it is ordered after the last use — which it is, because
    // stream already waits for all gate_done events before lstm_update_kernel.
    //
    // That approach is not used here because the performance difference is
    // negligible: cudaMallocAsync is a ~1–5 µs pool bump and the four
    // cudaStreamWaitEvent calls for setup_done cost a similar amount, while
    // each run_gate (two int8 GEMMs + pointwise activate) takes 100–500 µs.
    // The event overhead is 100× smaller than the work it guards, so the
    // saving would not be measurable. The current layout is kept because the
    // explicit setup_done barrier makes the ordering intent easier to follow.
    float *i_gate, *f_gate, *g_gate, *o_gate;
    cudaMallocAsync(&i_gate, sizeof(float) * B * H, stream);
    cudaMallocAsync(&f_gate, sizeof(float) * B * H, stream);
    cudaMallocAsync(&g_gate, sizeof(float) * B * H, stream);
    cudaMallocAsync(&o_gate, sizeof(float) * B * H, stream);

    // Two event fences per call:
    // setup_done event — recorded on the caller's stream after the 4
    // cudaMallocAsync calls. Each gate_streams_[i] waits on it before
    // launching. Without this, a gate stream could start writing to a buffer
    // that hasn't been allocated yet (the malloc is async, so there's no
    // implicit ordering between stream and the gate streams).
    // gate_done[4] events — each gate stream records one after its run_gate
    // completes. The caller's stream waits on all four before
    // lstm_update_kernel, which needs all four gate outputs ready. Without
    // this, the update kernel could race ahead of a still-running gate.
    // cudaEventDisableTiming is used on all events — it skips the hardware
    // timestamp, making the events cheaper to create and record.
    // cudaStreamNonBlocking on the gate streams prevents them from implicitly
    // synchronizing with the legacy NULL stream, so they stay truly independent.
    // The cudaFreeAsync calls at the end stay on stream, which is correct —
    // they're ordered after lstm_update_kernel by the stream's own queue.

    cudaEvent_t setup_done;
    cudaEventCreateWithFlags(&setup_done, cudaEventDisableTiming);

    // Plant flag
    cudaEventRecord(setup_done, stream);

    // Pause till flag is stamped
    for (int i = 0; i < 4; i++)
        cudaStreamWaitEvent(gate_streams_[i], setup_done, 0);
    cudaEventDestroy(setup_done);

    // The four gates are independent — dispatch them concurrently on
    // gate_streams_[0..3].  Each stream runs its own quantize + 2×GEMM +
    // activate pipeline without blocking the others.

    // i = sigmoid( ii(x) + hi(h) )
    run_gate<KernelType, BiasType>(
        x, h, ii_kernel, hi_kernel, hi_bias, i_gate,
        B, input_dim, H, /*use_sigmoid=*/true,
        x_quant_scale * ii_kernel_scale,
        h_quant_scale * hi_kernel_scale,
        h_quant_scale, gate_streams_[0]);

    // f = sigmoid( if(x) + hf(h) )
    run_gate<KernelType, BiasType>(
        x, h, if_kernel, hf_kernel, hf_bias, f_gate,
        B, input_dim, H, /*use_sigmoid=*/true,
        x_quant_scale * if_kernel_scale,
        h_quant_scale * hf_kernel_scale,
        h_quant_scale, gate_streams_[1]);

    // g = tanh( ig(x) + hg(h) )
    run_gate<KernelType, BiasType>(
        x, h, ig_kernel, hg_kernel, hg_bias, g_gate,
        B, input_dim, H, /*use_sigmoid=*/false,
        x_quant_scale * ig_kernel_scale,
        h_quant_scale * hg_kernel_scale,
        h_quant_scale, gate_streams_[2]);

    // o = sigmoid( io(x) + ho(h) )
    run_gate<KernelType, BiasType>(
        x, h, io_kernel, ho_kernel, ho_bias, o_gate,
        B, input_dim, H, /*use_sigmoid=*/true,
        x_quant_scale * io_kernel_scale,
        h_quant_scale * ho_kernel_scale,
        h_quant_scale, gate_streams_[3]);

    // Rejoin: make the caller's stream wait for all four gates to finish
    // before launching lstm_update_kernel, which reads all four gate outputs.
    
    cudaEvent_t gate_done[4];
    for (int i = 0; i < 4; i++) {
        // When you create a normal event (with timing), the GPU writes a
        // hardware timestamp into device memory every time cudaEventRecord is
        // called. That timestamp can later be read by cudaEventElapsedTime to
        // measure how long something took.
        // Writing that timestamp costs a small but real amount of work — a
        // global memory write that can stall the GPU's command processor
        // momentarily.
        // cudaEventDisableTiming tells the GPU: don't bother writing a
        // timestamp, just mark the event as reached. The event still works as
        // a synchronization fence — cudaStreamWaitEvent still correctly pauses a
        // stream until the event fires. It just can't be used for timing.
        cudaEventCreateWithFlags(&gate_done[i], cudaEventDisableTiming);
        cudaEventRecord(gate_done[i], gate_streams_[i]);
        cudaStreamWaitEvent(stream, gate_done[i], 0);
        cudaEventDestroy(gate_done[i]);
    }

    // new_c = f * c + i * g
    // new_h = o * tanh(new_c)
    // Both outputs stay in float32; no quantization of states.
    int total = B * H;
    int block = 256;
    int grid  = (total + block - 1) / block;
    lstm_update_kernel<<<grid, block, 0, stream>>>(
        c, i_gate, f_gate, g_gate, o_gate, new_c, new_h, total);

    cudaFreeAsync(i_gate, stream);
    cudaFreeAsync(f_gate, stream);
    cudaFreeAsync(g_gate, stream);
    cudaFreeAsync(o_gate, stream);
}


template class LSTMCell<int8_t, int32_t>;
