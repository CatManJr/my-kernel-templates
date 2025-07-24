#include <cuda_runtime.h>


__global__ void convolution_2d_kernel(const float* input, const float* kernel, float* output,
                                      int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    __shared__ float shared_kernel[1024];//more than 31*31
    __shared__ float shared_input[2304];

    const int output_rows = input_rows - kernel_rows + 1;
    const int output_cols = input_cols - kernel_cols + 1;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int shared_input_rows = blockDim.x + kernel_rows - 1;//current block can only reach to max(threadIdx.x+kidx)_rd row, which is blockDim.x+kernel_rows
    const int shared_input_cols = blockDim.y + kernel_cols - 1;
    const int input_length = input_rows*input_cols;

    if(threadIdx.x < kernel_rows && threadIdx.y < kernel_cols) {
        shared_kernel[threadIdx.x*kernel_cols + threadIdx.y] = kernel[threadIdx.x*kernel_cols + threadIdx.y];
    }
    const int num_threads = blockDim.x * blockDim.y;// num of threads available in this block
    const int shared_input_length = shared_input_rows*shared_input_cols; // num of shared inputs which should be copied
    const int num_iter = (shared_input_length + num_threads - 1) / num_threads;
    for(int i=0; i < num_iter; ++i) {
        const int shared_tid = num_threads*i + threadIdx.x*blockDim.y + threadIdx.y;
        const int shared_tidx = shared_tid / shared_input_cols;
        const int shared_tidy = shared_tid - shared_tidx*shared_input_cols;
        const int idx = shared_tidx + blockIdx.x*blockDim.x;
        const int idy = shared_tidy + blockIdx.y*blockDim.y; 
        const int input_id = idx*input_cols+idy;
        if(input_id < input_length) {
            shared_input[shared_tid] = input[input_id];
        }
        //memory is sufficient because num_iter=((31+16-1)*(31+16-1)+16*16-1)/16/16=9 in worst case and we need 9*16*16 memory
    }

    __syncthreads();

    if(idx < output_rows && idy < output_cols) {
        float out = 0.0;
        for(int kidx = 0; kidx < kernel_rows; ++kidx) {
            for(int kidy = 0; kidy < kernel_cols; ++kidy) {
                out += shared_kernel[kidx*kernel_cols + kidy]*shared_input[(kidx+threadIdx.x)*shared_input_cols+kidy+threadIdx.y];
            }
        }
        output[idx*output_cols + idy] = out;
    }
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output,
           int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    dim3 threadsPerBlock(16, 16);
    const int output_rows = input_rows - kernel_rows + 1;
    const int output_cols = input_cols - kernel_cols + 1;
    dim3 blocksPerGrid((output_rows + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (output_cols + threadsPerBlock.y - 1) / threadsPerBlock.y);

    convolution_2d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input,kernel,output,input_rows,input_cols,kernel_rows,kernel_cols);
    cudaDeviceSynchronize();
}