#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void prefix_sum_kernel(const float* input, float* output, int N) {
    constexpr int warp_size = 32;
    __shared__ float warp_last[1024 / warp_size]; // Assuming 1024 threads per block

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<warp_size> warp = cg::tiled_partition<warp_size>(block);
    int lane_id = warp.thread_rank();

    float val = (idx < N) ? input[idx] : 0.0f;

    for (int bucket_dim = 1; 2 * bucket_dim <= warp_size; bucket_dim *= 2) {
        int bucket_id = lane_id / bucket_dim;
        int prev_bucket_elem = bucket_id * bucket_dim - 1;
        int is_active_bucket = -(bucket_id % 2);
        val += __int_as_float(is_active_bucket & __float_as_int(warp.shfl(val, prev_bucket_elem)));
    }

    if (lane_id == warp_size - 1) {
        warp_last[threadIdx.x / warp_size] = val;
    }
    block.sync();

    for (int bucket_dim = warp_size; 2 * bucket_dim <= blockDim.x; bucket_dim *= 2) {
        int bucket_id = threadIdx.x / bucket_dim;
        if (threadIdx.x % bucket_dim == bucket_dim - 1) {
            val += warp_last[bucket_id - 1];
        }
        block.sync();
    }

    if (idx < N) {
        output[idx] = val;
    }
}

__global__ void block_merge_kernel(float* output, int N, int group_dim) {
    __shared__ float last_sum;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = idx / group_dim;
    int out_index = group_dim * (tid + 1) + idx % group_dim;

    if (threadIdx.x == 0) {
        last_sum = output[group_dim * (2 * tid + 1) - 1];
    }
    __syncthreads();

    if (out_index < N) {
        output[out_index] += last_sum;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    constexpr int threads_per_block = 512;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
    prefix_sum_kernel<<<blocks_per_grid, threads_per_block>>>(input, output, N);
    cudaDeviceSynchronize();

    for (int group_dim = threads_per_block; group_dim < N; group_dim *= 2) {
        int num_threads = group_dim * ((N + group_dim - 1) / group_dim / 2);
        if (num_threads == 0) break;
        int blocks_per_grid = (num_threads + threads_per_block - 1) / threads_per_block;
        block_merge_kernel<<<blocks_per_grid, threads_per_block>>>(output, N, group_dim);
        cudaDeviceSynchronize();
    }
}