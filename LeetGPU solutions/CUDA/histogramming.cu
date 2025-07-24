#include <cuda_runtime.h>

__global__ void histogramKernel(const int* input, int* histogram, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int val = input[idx];
        atomicAdd(&histogram[val], 1);
    }
}

extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {
    // Initialize the histogram array to zero
    cudaMemset(histogram, 0, num_bins * sizeof(int));

    // Compute grid and block dimensions
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Launch the kernel to compute the histogram
    histogramKernel<<<numBlocks, blockSize>>>(input, histogram, N);
}
