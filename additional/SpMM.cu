/* 
    Running in FUNCTIONAL mode...
    Compiling...
    Executing...
    BSR SpMM took 7.17247 ms
    Naive MM took 34.0208 ms
    Test failed!
    Exit status: 0
*/

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <iomanip>

struct BSRMatrix {
    float* values;      // Values of the non-zero blocks
    int* row_indices;   // Row indices of the non-zero blocks
    int* col_indices;   // Column indices of the non-zero blocks
    int* row_ptr;       // Pointer to the start of each row
    int M;              // Number of rows in the matrix (in blocks)
    int N;              // Number of columns in the matrix (in blocks)
    int nnz;            // Number of non-zero blocks
    int block_size;     // Size of the sub-blocks
};

// BSR sparse matrix multiplication kernel
__global__ void bsr_matrix_matrix_multiplication(const BSRMatrix A, const BSRMatrix B, BSRMatrix C) {
    int block_row = blockIdx.x;
    int block_col = blockIdx.y;

    if (block_row >= A.M || block_col >= B.N) return;

    int row_start = A.row_ptr[block_row];
    int row_end = A.row_ptr[block_row + 1];

    float result[4] = {0.0f};

    for (int i = row_start; i < row_end; ++i) {
        int a_block_col = A.col_indices[i];
        const float* a_block = &A.values[i * 4];

        int b_row_start = B.row_ptr[a_block_col];
        int b_row_end = B.row_ptr[a_block_col + 1];

        for (int j = b_row_start; j < b_row_end; ++j) {
            int b_block_col = B.col_indices[j];
            const float* b_block = &B.values[j * 4];

            result[0] += a_block[0] * b_block[0] + a_block[1] * b_block[2];
            result[1] += a_block[0] * b_block[1] + a_block[1] * b_block[3];
            result[2] += a_block[2] * b_block[0] + a_block[3] * b_block[2];
            result[3] += a_block[2] * b_block[1] + a_block[3] * b_block[3];
        }
    }

    int c_index = C.row_ptr[block_row] + block_col;
    if (C.col_indices[c_index] == block_col) {
        float* c_block = &C.values[c_index * 4];
        c_block[0] = result[0];
        c_block[1] = result[1];
        c_block[2] = result[2];
        c_block[3] = result[3];
    }
}

// Naive matrix multiplication kernel
__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}

// Convert BSR matrix to dense matrix
__global__ void bsr_to_dense(const BSRMatrix bsr_mat, float* dense_mat) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < bsr_mat.M * bsr_mat.block_size && col < bsr_mat.N * bsr_mat.block_size) {
        int block_row = row / bsr_mat.block_size;
        int block_col = col / bsr_mat.block_size;
        int local_row = row % bsr_mat.block_size;
        int local_col = col % bsr_mat.block_size;

        int nnz_index = bsr_mat.row_ptr[block_row] + block_col;
        if (nnz_index < bsr_mat.nnz && bsr_mat.col_indices[nnz_index] == block_col) {
            dense_mat[row * bsr_mat.N * bsr_mat.block_size + col] = bsr_mat.values[nnz_index * 4 + local_row * 2 + local_col];
        } else {
            dense_mat[row * bsr_mat.N * bsr_mat.block_size + col] = 0.0f;
        }
    }
}

void solve(const BSRMatrix A, const BSRMatrix B, BSRMatrix C) {
    dim3 gridDim(A.M, B.N);
    dim3 blockDim(1, 1);
    bsr_matrix_matrix_multiplication<<<gridDim, blockDim>>>(A, B, C);
    cudaDeviceSynchronize();
}

void naive_solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 blockDim(16, 16);
    dim3 gridDim((K + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
    matrix_multiplication_kernel<<<gridDim, blockDim>>>(A, B, C, M * 2, N * 2, K * 2);
    cudaDeviceSynchronize();
}

int main() {
    // Matrix dimensions and sparsity parameters
    int M = 100;  // Number of rows of matrix A (in blocks)
    int N = 100;  // Number of columns of matrix A (in blocks) and rows of matrix B (in blocks)
    int K = 100;  // Number of columns of matrix B (in blocks)
    int block_size = 2;  // Sub-block size
    int nnz_A = 1000;  // Number of non-zero blocks in matrix A
    int nnz_B = 1000;  // Number of non-zero blocks in matrix B

    // Generate random BSR matrices
    BSRMatrix h_A, h_B, h_C;
    // In a real application, generate random BSR matrices based on sparsity
    h_A.values = new float[nnz_A * block_size * block_size];
    h_A.row_indices = new int[nnz_A];
    h_A.col_indices = new int[nnz_A];
    h_A.row_ptr = new int[M + 1];
    h_A.M = M;
    h_A.N = N;
    h_A.nnz = nnz_A;
    h_A.block_size = block_size;

    h_B.values = new float[nnz_B * block_size * block_size];
    h_B.row_indices = new int[nnz_B];
    h_B.col_indices = new int[nnz_B];
    h_B.row_ptr = new int[N + 1];
    h_B.M = N;
    h_B.N = K;
    h_B.nnz = nnz_B;
    h_B.block_size = block_size;

    h_C.values = new float[M * K * block_size * block_size];
    h_C.row_indices = new int[M * K];
    h_C.col_indices = new int[M * K];
    h_C.row_ptr = new int[M + 1];
    h_C.M = M;
    h_C.N = K;
    h_C.nnz = M * K;
    h_C.block_size = block_size;

    // Initialize with sample data
    for (int i = 0; i < nnz_A * block_size * block_size; ++i) {
        h_A.values[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < nnz_A; ++i) {
        h_A.row_indices[i] = i / (N / block_size);
        h_A.col_indices[i] = i % (N / block_size);
    }
    for (int i = 0; i <= M; ++i) {
        h_A.row_ptr[i] = i * (nnz_A / M);
    }

    for (int i = 0; i < nnz_B * block_size * block_size; ++i) {
        h_B.values[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < nnz_B; ++i) {
        h_B.row_indices[i] = i / (K / block_size);
        h_B.col_indices[i] = i % (K / block_size);
    }
    for (int i = 0; i <= N; ++i) {
        h_B.row_ptr[i] = i * (nnz_B / N);
    }

    for (int i = 0; i < M * K * block_size * block_size; ++i) {
        h_C.values[i] = 0.0f;
    }
    for (int i = 0; i < M * K; ++i) {
        h_C.row_indices[i] = i / (K / block_size);
        h_C.col_indices[i] = i % (K / block_size);
    }
    for (int i = 0; i <= M; ++i) {
        h_C.row_ptr[i] = i * K;
    }

    // Copy matrices to the device
    BSRMatrix d_A, d_B, d_C;
    cudaMalloc(&d_A.values, h_A.nnz * block_size * block_size * sizeof(float));
    cudaMalloc(&d_A.row_indices, h_A.nnz * sizeof(int));
    cudaMalloc(&d_A.col_indices, h_A.nnz * sizeof(int));
    cudaMalloc(&d_A.row_ptr, (h_A.M + 1) * sizeof(int));
    cudaMemcpy(d_A.values, h_A.values, h_A.nnz * block_size * block_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A.row_indices, h_A.row_indices, h_A.nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A.col_indices, h_A.col_indices, h_A.nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A.row_ptr, h_A.row_ptr, (h_A.M + 1) * sizeof(int), cudaMemcpyHostToDevice);
    d_A.M = h_A.M;
    d_A.N = h_A.N;
    d_A.nnz = h_A.nnz;
    d_A.block_size = h_A.block_size;

    cudaMalloc(&d_B.values, h_B.nnz * block_size * block_size * sizeof(float));
    cudaMalloc(&d_B.row_indices, h_B.nnz * sizeof(int));
    cudaMalloc(&d_B.col_indices, h_B.nnz * sizeof(int));
    cudaMalloc(&d_B.row_ptr, (h_B.M + 1) * sizeof(int));
    cudaMemcpy(d_B.values, h_B.values, h_B.nnz * block_size * block_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B.row_indices, h_B.row_indices, h_B.nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B.col_indices, h_B.col_indices, h_B.nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B.row_ptr, h_B.row_ptr, (h_B.M + 1) * sizeof(int), cudaMemcpyHostToDevice);
    d_B.M = h_B.M;
    d_B.N = h_B.N;
    d_B.nnz = h_B.nnz;
    d_B.block_size = h_B.block_size;

    cudaMalloc(&d_C.values, h_C.nnz * block_size * block_size * sizeof(float));
    cudaMalloc(&d_C.row_indices, h_C.nnz * sizeof(int));
    cudaMalloc(&d_C.col_indices, h_C.nnz * sizeof(int));
    cudaMalloc(&d_C.row_ptr, (h_C.M + 1) * sizeof(int));
    cudaMemcpy(d_C.values, h_C.values, h_C.nnz * block_size * block_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C.row_indices, h_C.row_indices, h_C.nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C.col_indices, h_C.col_indices, h_C.nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C.row_ptr, h_C.row_ptr, (h_C.M + 1) * sizeof(int), cudaMemcpyHostToDevice);
    d_C.M = h_C.M;
    d_C.N = h_C.N;
    d_C.nnz = h_C.nnz;
    d_C.block_size = h_C.block_size;

    // Convert BSR matrices to dense matrices
    float* d_A_dense, *d_B_dense, *d_C_dense, *d_C_naive;
    int dense_size = M * K * block_size * block_size;
    cudaMalloc(&d_A_dense, dense_size * sizeof(float));
    cudaMalloc(&d_B_dense, dense_size * sizeof(float));
    cudaMalloc(&d_C_dense, dense_size * sizeof(float));
    cudaMalloc(&d_C_naive, dense_size * sizeof(float));

    dim3 blockDim(16, 16);
    dim3 gridDim((K * block_size + blockDim.x - 1) / blockDim.x, (M * block_size + blockDim.y - 1) / blockDim.y);
    bsr_to_dense<<<gridDim, blockDim>>>(d_A, d_A_dense);
    bsr_to_dense<<<gridDim, blockDim>>>(d_B, d_B_dense);

    // Record start time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Execute sparse matrix multiplication
    solve(d_A, d_B, d_C);

    // Convert BSR result to dense matrix
    bsr_to_dense<<<gridDim, blockDim>>>(d_C, d_C_dense);

    // Record end time and calculate elapsed time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "BSR SpMM took " << milliseconds << " ms" << std::endl;

    // Record start time
    cudaEventRecord(start);

    // Execute naive matrix multiplication
    naive_solve(d_A_dense, d_B_dense, d_C_naive, M, N, K);

    // Record end time and calculate elapsed time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Naive MM took " << milliseconds << " ms" << std::endl;

    // Compare results
    float* h_C_dense = new float[dense_size];
    float* h_C_naive = new float[dense_size];
    cudaMemcpy(h_C_dense, d_C_dense, dense_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_naive, d_C_naive, dense_size * sizeof(float), cudaMemcpyDeviceToHost);

    bool result_correct = true;
    for (int i = 0; i < dense_size; ++i) {
        if (abs(h_C_dense[i] - h_C_naive[i]) > 1e-5) {
            result_correct = false;
            break;
        }
    }

    if (result_correct) {
        std::cout << "Test passed!" << std::endl;
    } else {
        std::cout << "Test failed!" << std::endl;
    }

    // Free device memory
    cudaFree(d_A.values);
    cudaFree(d_A.row_indices);
    cudaFree(d_A.col_indices);
    cudaFree(d_A.row_ptr);
    cudaFree(d_B.values);
    cudaFree(d_B.row_indices);
    cudaFree(d_B.col_indices);
    cudaFree(d_B.row_ptr);
    cudaFree(d_C.values);
    cudaFree(d_C.row_indices);
    cudaFree(d_C.col_indices);
    cudaFree(d_C.row_ptr);
    cudaFree(d_A_dense);
    cudaFree(d_B_dense);
    cudaFree(d_C_dense);
    cudaFree(d_C_naive);

    delete[] h_A.values;
    delete[] h_A.row_indices;
    delete[] h_A.col_indices;
    delete[] h_A.row_ptr;
    delete[] h_B.values;
    delete[] h_B.row_indices;
    delete[] h_B.col_indices;
    delete[] h_B.row_ptr;
    delete[] h_C.values;
    delete[] h_C.row_indices;
    delete[] h_C.col_indices;
    delete[] h_C.row_ptr;
    delete[] h_C_dense;
    delete[] h_C_naive;

    return 0;
}