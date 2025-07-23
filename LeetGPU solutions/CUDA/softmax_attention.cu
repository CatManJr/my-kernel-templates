/*
    This solution provides both a standard softmax implementation and an online softmax version.
    To check online softmax and warp_shuffle plz jump into online_softmax_hopper.cu
*/

#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <float.h>
#define TILE_DIM 32

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// Q (M, d), K (N, d) -> scores (M, N)
__global__ void matmul_T(const float* __restrict__ Q, const float* __restrict__ K, float* scores, int M, int N, int d) {
    __shared__ float shared_Q[TILE_DIM][TILE_DIM];
    __shared__ float shared_K[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float sum = 0.0f;
    float scale = rsqrtf((float)d);  // faster than sqrtf(d)

    // Loop over tiles
    for (int t = 0; t < (d + TILE_DIM - 1) / TILE_DIM; ++t) {
        // Load Q tile from global to shared memory
        if (row < M && (t * TILE_DIM + threadIdx.x) < d)
            shared_Q[threadIdx.y][threadIdx.x] = Q[row * d + t * TILE_DIM + threadIdx.x];
        else
            shared_Q[threadIdx.y][threadIdx.x] = 0.0f;

        // Load K tile from global to shared memory (transposed access)
        if (col < N && (t * TILE_DIM + threadIdx.y) < d)
            shared_K[threadIdx.y][threadIdx.x] = K[col * d + t * TILE_DIM + threadIdx.y];
        else
            shared_K[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial dot product
        for (int i = 0; i < TILE_DIM; ++i) {
            sum += shared_Q[threadIdx.y][i] * shared_K[i][threadIdx.x];
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        scores[row * N + col] = sum * scale;
    }
}

// Softmax along rows of scores (M, N)
__global__ void softmax(float* scores, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        float max_val = -INFINITY;
        for (int j = 0; j < N; ++j) {
            max_val = fmaxf(max_val, scores[row * N + j]);
        }

        float sum_exp = 0.0f;
        for (int j = 0; j < N; ++j) {
            scores[row * N + j] = expf(scores[row * N + j] - max_val);
            sum_exp += scores[row * N + j];
        }

        sum_exp = fmaxf(sum_exp, 1e-8f);  // Avoid division by zero

        for (int j = 0; j < N; ++j) {
            scores[row * N + j] /= sum_exp;
        }
    }
}

// Online-Softmax + warp-shuffle by rows for hopper
__global__ void softmax_online(float* scores, int M, int N) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    float max_val = -FLT_MAX;
    float sum_exp = 0.0f;

    // single traverse
    for (int col = 0; col < N; ++col) {
        float val = scores[row * N + col];
        float new_max = fmaxf(max_val, val);
        sum_exp = sum_exp * __expf(max_val - new_max) + __expf(val - new_max);
        max_val = new_max;
    }

    // normalize
    for (int col = 0; col < N; ++col) {
        scores[row * N + col] = __expf(scores[row * N + col] - max_val) / sum_exp;
    }
}

// scores (M, N) x V (N, d) -> output (M, d)
__global__ void matmul(const float* __restrict__ scores, const float* __restrict__ V, float* output, int M, int N, int d) {
    __shared__ float shared_scores[TILE_DIM][TILE_DIM];
    __shared__ float shared_V[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles along the N dimension
    for (int t = 0; t < (N + TILE_DIM - 1) / TILE_DIM; ++t) {
        // Load a tile from scores (M x N)
        if (row < M && t * TILE_DIM + threadIdx.x < N)
            shared_scores[threadIdx.y][threadIdx.x] = scores[row * N + t * TILE_DIM + threadIdx.x];
        else
            shared_scores[threadIdx.y][threadIdx.x] = 0.0f;

        // Load a tile from V (N x d)
        if (col < d && t * TILE_DIM + threadIdx.y < N)
            shared_V[threadIdx.y][threadIdx.x] = V[(t * TILE_DIM + threadIdx.y) * d + col];
        else
            shared_V[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial dot product
        for (int i = 0; i < TILE_DIM; ++i) {
            sum += shared_scores[threadIdx.y][i] * shared_V[i][threadIdx.x];
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < d) {
        output[row * d + col] = sum;
    }
}

// device 2 host
void print_matrix(const float* d_mat, int row, int col) {
    float* h_mat = new float[row * col];
    CUDA_CHECK(cudaMemcpy(h_mat, d_mat, row * col * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            std::cout << std::fixed << std::setprecision(4) << h_mat[i * col + j] << " ";
        }
        std::cout << std::endl;
    }

    delete[] h_mat;
}

// built up
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    float* scores;
    CUDA_CHECK(cudaMalloc(&scores, M * N * sizeof(float)));
    CUDA_CHECK(cudaMemset(scores, 0, M * N * sizeof(float)));

    // --- Step 1: scores = Q * K^T ---
    dim3 blockSizeMat(TILE_DIM, TILE_DIM);
    dim3 gridSizeMat((N + blockSizeMat.x - 1) / blockSizeMat.x,
                     (M + blockSizeMat.y - 1) / blockSizeMat.y);

    matmul_T<<<gridSizeMat, blockSizeMat>>>(Q, K, scores, M, N, d);
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Step 2: softmax(scores) ---
    int threads = 256;
    int blocks = (M + threads - 1) / threads;
    softmax_online<<<blocks, threads>>>(scores, M, N);
    // softmax<<<blocks, threads>>>(scores, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Step 3: output = scores * V ---
    matmul<<<gridSizeMat, blockSizeMat>>>(scores, V, output, M, N, d);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(scores));
}