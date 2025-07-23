#include <cuda_runtime.h>
#include <float.h>

#define WARP_SIZE 32

// ---------- Helper struct ----------
struct __align__(8) MD {
    float m;   // running max
    float d;   // running sum(exp(x - m))
};

// ---------- Warp-level online softmax reduction ----------
template <int kWarpSize = WARP_SIZE>
__device__ __forceinline__ MD warp_reduce_md_op(MD val) {
    unsigned mask = 0xffffffffu;
#pragma unroll
    for (int stride = kWarpSize >> 1; stride >= 1; stride >>= 1) {
        MD other;
        other.m = __shfl_xor_sync(mask, val.m, stride);
        other.d = __shfl_xor_sync(mask, val.d, stride);

        bool bigger = val.m > other.m;
        float new_m = bigger ? val.m : other.m;
        val.d = (bigger ? val.d : other.d) +
                (bigger ? other.d : val.d) * __expf((bigger ? other.m : val.m) - new_m);
        val.m = new_m;
    }
    return val;
}

__global__ void online_big_softmax_kernel(const float* __restrict__ x,
                                          float* __restrict__ y, int N)
{
    constexpr int NUM_THREADS = 1024;
    const int tid  = threadIdx.x;
    const int wid  = tid / 32;
    const int lane = tid % 32;
    constexpr int WARPS = NUM_THREADS / 32;

    __shared__ MD smem[WARPS];

    // 1) Global online reduce: each thread sequentially processes several elements
    MD global{-FLT_MAX, 0.0f};
    for (int i = tid; i < N; i += NUM_THREADS) {
        float v = x[i];
        float old_m = global.m;
        global.m = fmaxf(global.m, v);
        global.d = global.d * __expf(old_m - global.m) +
                   __expf(v - global.m);
    }

    // 2) Intra-warp reduction
    global = warp_reduce_md_op<32>(global);

    // 3) Reduce the 32 warps within the block
    if (lane == 0) smem[wid] = global;
    __syncthreads();

    if (tid < 32) {
        MD val = (tid < WARPS) ? smem[tid] : MD{-FLT_MAX, 0.0f};
        val = warp_reduce_md_op<WARPS>(val);
        if (tid == 0) smem[0] = val;
    }
    __syncthreads();
    MD final = smem[0];

    // 4) Write back the results
    float inv = __fdividef(1.0f, final.d);
    for (int i = tid; i < N; i += NUM_THREADS)
        y[i] = __expf(x[i] - final.m) * inv;
}

extern "C" void solve(const float* input, float* output, int N) {
    online_big_softmax_kernel<<<1, 1024>>>(input, output, N);
    cudaDeviceSynchronize();
}