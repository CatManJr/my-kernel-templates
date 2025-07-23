#include <cuda_runtime.h>

const int BLOCK_SIZE = 256;

// Reduction kernel
__global__ void reduction_kernel(const float* input, float* output, int N) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load input into shared memory
    float mySum = 0.0f;
    if (i < N) {
        mySum = input[i];
    }
    
    // Reduce within a block
    sdata[tid] = mySum;
    __syncthreads();
    
    // Unroll the reduction loop for better performance
    if (BLOCK_SIZE >= 512 && tid < 256) { sdata[tid] = mySum = mySum + sdata[tid + 256]; }
    __syncthreads();
    if (BLOCK_SIZE >= 256 && tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; }
    __syncthreads();
    if (BLOCK_SIZE >= 128 && tid < 64) { sdata[tid] = mySum = mySum + sdata[tid + 64]; }
    __syncthreads();
    
    // Warp-level reduction
    if (tid < 32) {
        volatile float* vsdata = sdata;
        vsdata[tid] = mySum = mySum + vsdata[tid + 32];
        vsdata[tid] = mySum = mySum + vsdata[tid + 16];
        vsdata[tid] = mySum = mySum + vsdata[tid + 8];
        vsdata[tid] = mySum = mySum + vsdata[tid + 4];
        vsdata[tid] = mySum = mySum + vsdata[tid + 2];
        vsdata[tid] = mySum = mySum + vsdata[tid + 1];
    }
    
    // Write result for this block to global memory
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float* temp_output;
    cudaMalloc(&temp_output, num_blocks * sizeof(float));
    
    // First reduction
    reduction_kernel<<<num_blocks, BLOCK_SIZE>>>(input, temp_output, N);
    
    // Recursive reduction if necessary
    while (num_blocks > 1) {
        int new_num_blocks = (num_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE;
        reduction_kernel<<<new_num_blocks, BLOCK_SIZE>>>(temp_output, temp_output, num_blocks);
        num_blocks = new_num_blocks;
    }
    
    // Copy the final result to the output
    cudaMemcpy(output, temp_output, sizeof(float), cudaMemcpyDeviceToDevice);
    
    // Free temporary memory
    cudaFree(temp_output);
    
    // Synchronize to ensure all operations are complete
    cudaDeviceSynchronize();
}