// nnarek@LeetGPU
#include <cuda_runtime.h>

constexpr int threadsPerBlock = 256;

__device__ inline void swap(float& a,float& b) {
    auto temp = a;
    a = b;
    b = temp;
}

__device__ inline void sort2(float& a,float& b) {
    if(b < a) {
        swap(a,b);
    }
}

__global__ void bitonic_block_sort_kernel(float* data, int N) {
    __shared__ float sdata[2*threadsPerBlock];

    const int tid = threadIdx.x;
    const int idx = (blockIdx.x * 2 * threadsPerBlock) + threadIdx.x;
    if(idx < N) {
        sdata[tid] = data[idx];
    } else {
        sdata[tid] = INFINITY;
    }
    if(idx + threadsPerBlock < N) {
        sdata[tid + threadsPerBlock] = data[idx + threadsPerBlock];
    } else {
        sdata[tid + threadsPerBlock] = INFINITY;
    }
    __syncthreads();

    for(int threads_in_chunk = 1; threads_in_chunk <= threadsPerBlock; threads_in_chunk *= 2) {
        const int items_in_chunk = 2*threads_in_chunk;
        const int chunk_id = tid/threads_in_chunk;
        const int tid_in_chunk = tid % threads_in_chunk;
        sort2(sdata[items_in_chunk*chunk_id+tid_in_chunk],
              sdata[items_in_chunk*chunk_id+items_in_chunk-1-tid_in_chunk]);//triangle layer
        __syncthreads();
        for(int threads_in_rhombus = threads_in_chunk/2; threads_in_rhombus > 0; threads_in_rhombus/=2) {
            const int items_in_rhombus = 2*threads_in_rhombus;
            const int rhombus_id = tid/threads_in_rhombus;
            const int tid_in_rhombus = tid % threads_in_rhombus;
            sort2(sdata[items_in_rhombus*rhombus_id+tid_in_rhombus],
                  sdata[items_in_rhombus*rhombus_id+threads_in_rhombus+tid_in_rhombus]);//rhombus layers
            __syncthreads();
        }
    }

    if(idx < N) {
        data[idx] = sdata[tid];
    }
    if(idx + threadsPerBlock < N) {
        data[idx + threadsPerBlock] = sdata[tid + threadsPerBlock];
    }
}

__global__ void bitonic_grid_sort_triangle_layer(float* data, int N, int threads_in_chunk) {
    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    const int items_in_chunk = 2*threads_in_chunk;
    const int chunk_id = tid/threads_in_chunk;
    const int tid_in_chunk = tid % threads_in_chunk;
    if((items_in_chunk*chunk_id+items_in_chunk-1-tid_in_chunk) < N) {
        sort2(data[items_in_chunk*chunk_id+tid_in_chunk],
              data[items_in_chunk*chunk_id+items_in_chunk-1-tid_in_chunk]);//triangle layer
    }
}

__global__ void bitonic_grid_sort_rhombus_layer(float* data, int N, int threads_in_rhombus) {
    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    const int items_in_rhombus = 2*threads_in_rhombus;
    const int rhombus_id = tid/threads_in_rhombus;
    const int tid_in_rhombus = tid % threads_in_rhombus;
    if((items_in_rhombus*rhombus_id+threads_in_rhombus+tid_in_rhombus) < N) {
        sort2(data[items_in_rhombus*rhombus_id+tid_in_rhombus],
                data[items_in_rhombus*rhombus_id+threads_in_rhombus+tid_in_rhombus]);//rhombus layers
    }
}

extern "C" void solve(float* data, int N) {
    int itemsPerBlock = 2 * threadsPerBlock;
    int blocksPerGrid = (N+itemsPerBlock-1) / itemsPerBlock;
    bitonic_block_sort_kernel<<<blocksPerGrid, threadsPerBlock>>>(data, N);
    cudaDeviceSynchronize();

    // when first time threads_in_chunk >= N then its mean that in previous iteration, number of threads is threads_in_chunk/2 
    // and we handled 2*threads_in_chunk/2 elements, which is >=N, hence we should stop here
    for(int threads_in_chunk = 2*threadsPerBlock; threads_in_chunk < N; threads_in_chunk *= 2) {
        // not optimal number of threads, because in most cases first threads of first chunk will do nothing
        // but we still need to create them because in bitonic_grid_sort_triangle_layer function 
        // we do not count which thread is actual first thread which will do some work 
        int numElements = threads_in_chunk * ((N + threads_in_chunk - 1) / threads_in_chunk);
        int blocksPerGrid = (numElements+threadsPerBlock-1) / threadsPerBlock;
        bitonic_grid_sort_triangle_layer<<<blocksPerGrid, threadsPerBlock>>>(data,N,threads_in_chunk);
        cudaDeviceSynchronize();
        for(int threads_in_rhombus = threads_in_chunk/2; threads_in_rhombus > 0; threads_in_rhombus/=2) {
            bitonic_grid_sort_rhombus_layer<<<blocksPerGrid, threadsPerBlock>>>(data,N,threads_in_rhombus);
            cudaDeviceSynchronize();
        }
    }
}