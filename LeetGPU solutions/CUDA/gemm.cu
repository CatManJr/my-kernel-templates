// very rough version

#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void gemm_fp16_kernel(const __half* A, const __half* B, __half* C,
                                 int M, int N, int K,
                                 float alpha, float beta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // 0..M-1
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // 0..N-1

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a = __half2float(A[row * K + k]);
            float b = __half2float(B[k * N + col]);
            acc += a * b;
        }
        float c_old = __half2float(C[row * N + col]);
        float c_new = alpha * acc + beta * c_old;
        C[row * N + col] = __float2half(c_new);
    }
}

extern "C" void solve(const __half* A, const __half* B, __half* C,
                      int M, int N, int K,
                      float alpha, float beta) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (M + 15) / 16);
    gemm_fp16_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();
}