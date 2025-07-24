// Use warp shuffle & recursive reduction to compute dot product

#include <cuda_runtime.h>

__global__ void dot_product_kernel(const float* A, const float* B, float* partial_sums, int N) {
    extern __shared__ float shared_mem[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    if (tid < N) {
        sum = A[tid] * B[tid];
    }

    shared_mem[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            shared_mem[threadIdx.x] += shared_mem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partial_sums[blockIdx.x] = shared_mem[0];
    }
}

__global__ void reduce_kernel(float* partial_sums, int N) {
    extern __shared__ float shared_mem[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N) {
        shared_mem[threadIdx.x] = partial_sums[tid];
    } else {
        shared_mem[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            shared_mem[threadIdx.x] += shared_mem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partial_sums[blockIdx.x] = shared_mem[0];
    }
}

extern "C" void solve(const float* A, const float* B, float* result, int N) {
    const int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;

    float* partial_sums;
    cudaMalloc(&partial_sums, grid_size * sizeof(float));
    cudaMemset(result, 0, sizeof(float));

    dot_product_kernel<<<grid_size, block_size, block_size * sizeof(float)>>>(A, B, partial_sums, N);
    cudaDeviceSynchronize();

    while (grid_size > 1) {
        int new_grid_size = (grid_size + block_size - 1) / block_size;
        reduce_kernel<<<new_grid_size, block_size, block_size * sizeof(float)>>>(partial_sums, grid_size);
        cudaDeviceSynchronize();
        grid_size = new_grid_size;
    }

    if (grid_size == 1) {
        cudaMemcpy(result, partial_sums, sizeof(float), cudaMemcpyDeviceToDevice);
    }

    cudaFree(partial_sums);
}