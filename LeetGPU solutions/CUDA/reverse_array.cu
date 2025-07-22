#include <cuda_runtime.h>

__global__ void reverse_array(float* input, int N) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto mid = N / 2;
    if (idx < mid) {
        auto out_idx = N - idx - 1;
        auto temp = input[idx];
        input[idx] = input[out_idx];
        input[out_idx] = temp;
    }
}

// input is device pointer
extern "C" void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}