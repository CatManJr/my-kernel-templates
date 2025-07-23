#include <cuda_runtime.h>

// vanilla copy
__global__ void copy_matrix_kernel(const float* A, float* B, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * N) {
        int i = idx / N;
        int j = idx % N;
        B[i * N + j] = A[i * N + j];
    }
}

// copy from shared memory to global memory
__global__ void copy_matrix_kernel_sm(const float* A, float* B, int N) {
    __shared__ float sharedMem[256]; // shared memory for 256 elements
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // 1d index
    int sharedIdx = threadIdx.x;

    // each thread copies one element from global memory to shared memory
    if (idx < N * N) {
        sharedMem[sharedIdx] = A[idx];
    }

    __syncthreads(); // await

    // each thread copies from shared memory to global memory
    if (idx < N * N) {
        B[idx] = sharedMem[sharedIdx];
    }
}

// A, B are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, float* B, int N) {
    int total = N * N;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;
    copy_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, N);
    cudaDeviceSynchronize();
} 