#include <cuda_runtime.h>
#include <float.h>

// Device-side utility: fast expf wrapper
__device__ __forceinline__ float my_exp(float x) { return expf(x); }

// Step 1: per-block max reduction
__global__ void max_reduction_kernel(const float* __restrict__ input,
                                     float* __restrict__ block_max,
                                     int N) {
    __shared__ float sdata[256];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;

    // load with safe fallback
    float val = (idx < N) ? input[idx] : -FLT_MAX;
    sdata[tid] = val;
    __syncthreads();

    // tree reduction
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        __syncthreads();
    }
    if (tid == 0) block_max[blockIdx.x] = sdata[0];
}

// Step 2: reduce partial maxima to global max
__global__ void final_max_kernel(const float* __restrict__ block_max,
                                 float* __restrict__ global_max,
                                 int num_blocks) {
    __shared__ float sdata[256];

    unsigned int tid = threadIdx.x;
    float val = -FLT_MAX;
    if (tid < num_blocks) val = block_max[tid];
    sdata[tid] = val;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        __syncthreads();
    }
    if (tid == 0) *global_max = sdata[0];
}

// Step 3: compute exp(x - max) and per-block sum
__global__ void exp_sum_kernel(const float* __restrict__ input,
                               float* __restrict__ output,
                               const float* __restrict__ max_val,
                               float* __restrict__ block_sum,
                               int N) {
    __shared__ float sdata[256];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;

    float val = 0.0f;
    if (idx < N) {
        val = my_exp(input[idx] - *max_val);
        output[idx] = val;          // store intermediate exp(x-max)
    }
    sdata[tid] = val;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    if (tid == 0) block_sum[blockIdx.x] = sdata[0];
}

// Step 4: normalize to produce softmax
__global__ void final_sum_and_divide_kernel(const float* __restrict__ output,
                                            const float* __restrict__ global_sum,
                                            float* __restrict__ out,
                                            int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) out[idx] = output[idx] / *global_sum;
}

// Entry point
extern "C" void solve(const float* input, float* output, int N) {
    const int threads = 256;
    int blocks = (N + threads - 1) / threads;

    float *d_block_max, *d_block_sum, *d_max, *d_sum;
    cudaMalloc(&d_block_max, blocks * sizeof(float));
    cudaMalloc(&d_block_sum, blocks * sizeof(float));
    cudaMalloc(&d_max,        sizeof(float));
    cudaMalloc(&d_sum,        sizeof(float));

    // 1. global max
    max_reduction_kernel<<<blocks, threads>>>(input, d_block_max, N);
    final_max_kernel<<<1, threads>>>(d_block_max, d_max, blocks);

    // 2. exp(x-max) and per-block sums
    exp_sum_kernel<<<blocks, threads>>>(input, output, d_max, d_block_sum, N);

    // 3. global sum
    final_max_kernel<<<1, threads>>>(d_block_sum, d_sum, blocks);

    // 4. softmax = exp(x-max) / sum
    final_sum_and_divide_kernel<<<blocks, threads>>>(output, d_sum, output, N);

    cudaFree(d_block_max);
    cudaFree(d_block_sum);
    cudaFree(d_max);
    cudaFree(d_sum);
    cudaDeviceSynchronize();
}