#include <cuda_runtime.h>

const int BLOCK_SIZE = 256;

// Generic reduction kernel
template <unsigned int blockSize>
__global__ void reduction_kernel(const float* input, float* output, int N) {
    __shared__ float sdata[blockSize];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data from global memory to shared memory
    sdata[tid] = (i < N) ? input[i] : 0.0f;
    __syncthreads();
    
    // Perform reduction in shared memory (unrolled loop)
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    
    // Use warp-level reduction
    if (tid < 32) {
        volatile float* vsdata = sdata;
        vsdata[tid] += vsdata[tid + 32];
        vsdata[tid] += vsdata[tid + 16];
        vsdata[tid] += vsdata[tid + 8];
        vsdata[tid] += vsdata[tid + 4];
        vsdata[tid] += vsdata[tid + 2];
        vsdata[tid] += vsdata[tid + 1];
    }
    
    // The first thread writes the result of this block
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Simple scaling kernel
__global__ void scale_kernel(float* input, float* output, float interval_size, int n_samples) {
    *output = (*input) * interval_size / n_samples;
}

// y_samples, result are device pointers
extern "C" void solve(const float* y_samples, float* result, float a, float b, int n_samples) {
    // Check for valid input
    if (n_samples <= 0) {
        cudaMemset(result, 0, sizeof(float));
        return;
    }
    
    // Calculate the number of thread blocks needed
    int num_blocks = (n_samples + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Allocate temporary storage
    float* temp_output;
    cudaMalloc(&temp_output, num_blocks * sizeof(float));
    
    // First reduction
    reduction_kernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE>>>(y_samples, temp_output, n_samples);
    
    // Multi-level reduction until only one result remains
    int remaining = num_blocks;
    while (remaining > 1) {
        int new_num_blocks = (remaining + BLOCK_SIZE - 1) / BLOCK_SIZE;
        reduction_kernel<BLOCK_SIZE><<<new_num_blocks, BLOCK_SIZE>>>(temp_output, temp_output, remaining);
        remaining = new_num_blocks;
    }
    
    // Copy the final result to the output
    cudaMemcpy(result, temp_output, sizeof(float), cudaMemcpyDeviceToDevice);
    
    // Free temporary memory
    cudaFree(temp_output);
    
    // Scale the final result
    scale_kernel<<<1, 1>>>(result, result, b - a, n_samples);
    
    // Synchronize the device
    cudaDeviceSynchronize();
}