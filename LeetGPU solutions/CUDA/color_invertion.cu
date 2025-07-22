#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;
    
    if (idx < total_pixels) {
        // Each pixel has 4 components (RGBA)
        int pixel_start = idx * 4;
        
        // Invert R, G, B but leave A unchanged
        image[pixel_start]     = 255 - image[pixel_start];     // R
        image[pixel_start + 1] = 255 - image[pixel_start + 1]; // G
        image[pixel_start + 2] = 255 - image[pixel_start + 2]; // B
        // Alpha (image[pixel_start + 3]) remains unchanged
    }
}
// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}