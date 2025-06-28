#include <stdio.h>
#include <cuda_runtime.h>

#define N 16

// Kernel definition
__global__ void MatAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    int j = threadIdx.y;
    int idx = i * N + j;
    C[idx] = A[idx] + B[idx];
}

int main()
{
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(float);
    
    // Allocate host memory
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);
    
    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int idx = i * N + j;
            h_A[idx] = i + j;
            h_B[idx] = i * j + 1;
        }
    }
    
    // Allocate device memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    // Copy host arrays to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // Kernel invocation with one block of N * N * 1 threads
    int numBlocks = 1;
    dim3 threadsPerBlock(N, N);
    MatAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);
    
    // Wait for GPU to finish
    cudaDeviceSynchronize();
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // Verify result
    printf("Matrix Addition Results (showing first 5x5 elements):\n");
    bool correct = true;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            int idx = i * N + j;
            float expected = h_A[idx] + h_B[idx];
            printf("C[%d][%d] = %.2f (expected: %.2f) ", i, j, h_C[idx], expected);
            if (h_C[idx] != expected) {
                correct = false;
            }
        }
        printf("\n");
    }
    
    // Check all results
    for (int i = 0; i < N * N; i++) {
        float expected = h_A[i] + h_B[i];
        if (h_C[i] != expected) {
            correct = false;
            break;
        }
    }
    
    printf("\nMatrix computation result: %s\n", correct ? "CORRECT" : "INCORRECT");
    
    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}