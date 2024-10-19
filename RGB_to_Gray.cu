#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stdio.h>
#include "stb_image.h"
#include "stb_image_write.h"

#define CHECK(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
} \


void rgbToGrayCPU(unsigned char *rgb, unsigned char *gray, int width, int height) {
for (int y = 0; y < height; y++) { // Loop over all rows of the image
for (int x = 0; x < width; x++) { // Loop over all pixels in a row
int rgbOffset = (y * width + x) * 3; // Calculate the offset for the RGB pixel
int grayOffset = y * width + x; // Calculate the offset for the grayscale pixel
unsigned char r = rgb[rgbOffset]; // Read the red value
unsigned char g = rgb[rgbOffset + 1]; // Read the green value
unsigned char b = rgb[rgbOffset + 2]; // Read the blue value
gray[grayOffset] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b); // RGB->Gray
}
}
}

__global__ void rgbToGrayGPU(unsigned char *d_rgb, unsigned char *d_gray, int width, int height) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x; // Calculate the x-coordinate of the pixel
    int iy = blockIdx.y * blockDim.y + threadIdx.y; // Calculate the y-coordinate of the pixel

    // Boundary check: ensures the thread is within the image
    if (ix < width && iy < height) {
        int rgbOffset = (iy * width + ix) * 3; // Calculate the offset for the RGB pixel
        int grayOffset = iy * width + ix; // Calculate the offset for the grayscale pixel

        unsigned char r = d_rgb[rgbOffset];     // Read the red value
        unsigned char g = d_rgb[rgbOffset + 1]; // Read the green value
        unsigned char b = d_rgb[rgbOffset + 2]; // Read the blue value

        // Convert RGB to grayscale
        d_gray[grayOffset] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

int main() {
    const char* imagePath = "/content/Prova1.png"; // Specifica qui il percorso dell'immagine
    // Load the image using "stb_image.h"
    int width, height, channels;
    unsigned char *rgb = stbi_load(imagePath, &width, &height, &channels, 3);
    if (!rgb) {
        printf("Error loading image %s\n", imagePath);
        return 1;
    }
    printf("Image loaded: %dx%d with %d channels\n", width, height, channels);

    // Allocate host memory for grayscale image
    int imageSize = width * height;
    int rgbSize = imageSize * 3;
    unsigned char *h_gray = (unsigned char *)malloc(imageSize); // Allocate memory for GPU output
    unsigned char *cpu_gray = (unsigned char *)malloc(imageSize); // Allocate memory for CPU output

    // Convert the image to grayscale on the CPU
    rgbToGrayCPU(rgb, cpu_gray, width, height);

    // Allocate device memory
    unsigned char *d_rgb, *d_gray;
    CHECK(cudaMalloc((void **)&d_rgb, rgbSize)); // Allocate memory for the RGB image on the GPU
    CHECK(cudaMalloc((void **)&d_gray, imageSize)); // Allocate memory for the grayscale output on the GPU

    // Copy data from host to device
    CHECK(cudaMemcpy(d_rgb, rgb, rgbSize, cudaMemcpyHostToDevice));

    // Configure and launch the CUDA kernel
    dim3 block(32, 32); // Block size: 32x32 threads
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    rgbToGrayGPU<<<grid, block>>>(d_rgb, d_gray, width, height); // Launch the kernel
    CHECK(cudaDeviceSynchronize()); // Wait for the kernel to finish

    // Copy the result from device to host
    CHECK(cudaMemcpy(h_gray, d_gray, imageSize, cudaMemcpyDeviceToHost));

    // Verify the result
    bool match = true;
    for (int i = 0; i < imageSize; i++) {
        if (abs(cpu_gray[i] - h_gray[i]) > 1) { // Allow a small difference due to rounding
            match = false;
            printf("Mismatch at pixel %d: CPU %d, GPU %d\n", i, cpu_gray[i], h_gray[i]);
            break;
        }
    }
    if (match) printf("CPU and GPU results match.\n");

    // Save the grayscale image
    stbi_write_png("output_gray.png", width, height, 1, h_gray, width);

    // Free memory
    stbi_image_free(rgb);
    free(h_gray);
    free(cpu_gray);
    CHECK(cudaFree(d_rgb));
    CHECK(cudaFree(d_gray));

    // Reset the CUDA device
    CHECK(cudaDeviceReset());

    return 0;
}
