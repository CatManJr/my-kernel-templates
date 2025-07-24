// nnarek@LeetGPU

#include <cuda_runtime.h>

struct item_t {
    float val;
    short row;
    short col;
};

__global__ void compress_mat_kernel(const float* mat, item_t* compressed_mat, int M, int N, int* last_index) {
    // Shared memory to count the number of non-zero elements in the current block
    __shared__ int numZerosInBlock;
    // Shared memory to store the starting index for non-zero elements in the compressed matrix
    __shared__ int index_start;

    // Calculate the global row and column indices for the current thread
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    int idy = (blockIdx.y * blockDim.y) + threadIdx.y;

    // Initialize the non-zero count for the block (only done by the first thread in each block)
    if(threadIdx.y == 0) {
        numZerosInBlock = 0;
    }
    __syncthreads(); // Ensure all threads see the initialized count

    // Check if the current thread is within the matrix bounds
    if(idx < M && idy < N) {
        const auto val = mat[idx * N + idy]; // Get the matrix value at (idx, idy)
        int local_index = -1;
        // If the value is non-zero, record its position within the block
        if(val != 0.0f) {
            local_index = atomicAdd(&numZerosInBlock, 1); // Atomically increment the non-zero count
        }
        __syncthreads(); // Ensure all threads have updated numZerosInBlock

        // The first thread in each block updates the global index in the compressed matrix
        if(threadIdx.y == 0) {
            index_start = atomicAdd(last_index, numZerosInBlock);
        }
        __syncthreads(); // Ensure all threads see the updated index_start

        // If the value is non-zero, store it in the compressed matrix
        if(local_index != -1) {
            compressed_mat[index_start + local_index] = {val, (short)idx, (short)idy};
        }
    }
}

__global__ void matvec_kernel(const item_t* compressed_mat, const float* v, float* y, int nnz) {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(idx < nnz) {
        auto item = compressed_mat[idx];
        atomicAdd(y + item.row, item.val*v[item.col]);
    }
}

extern "C" void solve(const float* mat, const float* v, float* y, int M, int N, int nnz) {
    item_t* compressed_mat;
    cudaMalloc(&compressed_mat, nnz * sizeof(item_t));
    int* last_index;
    cudaMalloc(&last_index, sizeof(int));
    cudaMemset(last_index, 0, sizeof(int));
    cudaMemset(y, 0, M * sizeof(float));
    {
        dim3 threadsPerBlock(1,256);
        dim3 blocksPerGrid((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                           (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
        compress_mat_kernel<<<blocksPerGrid, threadsPerBlock>>>(mat, compressed_mat, M, N, last_index);
    }
    {
        dim3 threadsPerBlock(256);
        dim3 blocksPerGrid((nnz + threadsPerBlock.x - 1) / threadsPerBlock.x);
        matvec_kernel<<<blocksPerGrid, threadsPerBlock>>>(compressed_mat, v, y, nnz);
    }
    cudaDeviceSynchronize();
    cudaFree(compressed_mat);
} 