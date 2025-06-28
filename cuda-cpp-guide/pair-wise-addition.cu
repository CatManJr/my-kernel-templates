#include <stdio.h>
#include <cuda_runtime.h>

#define N 256

// Kernel definition
__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main()
{
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    size_t size = N * sizeof(float);
    
    // Allocate host memory
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);
    
    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }
    
    // Allocate device memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    // Copy host arrays to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // Kernel invocation with N threads
    VecAdd<<<1, N>>>(d_A, d_B, d_C);
    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // Verify result
    printf("First few results:\n");
    bool correct = true;
    for (int i = 0; i < 10; i++) {
        printf("C[%d] = %.2f (expected: %.2f)\n", i, h_C[i], h_A[i] + h_B[i]);
        if (h_C[i] != h_A[i] + h_B[i]) {
            correct = false;
        }
    }
    
    // Check all results
    for (int i = 0; i < N; i++) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            correct = false;
            break;
        }
    }
    
    printf("\nComputation result: %s\n", correct ? "CORRECT" : "INCORRECT");
    
    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}