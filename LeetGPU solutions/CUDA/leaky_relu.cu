#include <cuda_runtime.h>
#define alpha 0.01f

__global__ void leaky_relu_kernel(const float* input, float* output, int N) {
    auto idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        output[idx] = input[idx] < 0 ? alpha * input[idx] : input[idx];
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    leaky_relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}