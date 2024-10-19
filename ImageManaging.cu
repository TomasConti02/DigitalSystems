//you have to download on the same path of the project the gitHub library stb_image.h/stb_image_write.h and the image that you want manage
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

__global__ void cudaImageFlip(unsigned char* input, unsigned char* output, int width, int height, int channels, bool horizontal) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x; // Calculate the x-coordinate of the pixel
    int iy = blockIdx.y * blockDim.y + threadIdx.y; // Calculate the y-coordinate of the pixel
    if (ix < width && iy < height) { // Check if the thread is within the image
        int outputIdx;
        int inputIdx = (iy * width + ix) * channels;
        
        if (horizontal) {
            outputIdx = (iy * width + (width - 1 - ix)) * channels; // Horizontal flip index
        } else {
            outputIdx = ((height - 1 - iy) * width + ix) * channels; // Vertical flip index
        }

        // Copy the pixel data for all channels
        for (int c = 0; c < channels; ++c) {
            output[outputIdx + c] = input[inputIdx + c];
        }
    }
}

__global__ void rgbToGrayGPU(unsigned char *d_rgb, unsigned char *d_gray, int width, int height) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x; // Calculate the x-coordinate of the pixel
    int iy = blockIdx.y * blockDim.y + threadIdx.y; // Calculate the y-coordinate of the pixel
    if (ix < width && iy < height) { // Ensure the thread is within the image
        int rgbOffset = (iy * width + ix) * 3; // Offset for the RGB pixel
        int grayOffset = iy * width + ix; // Offset for the grayscale pixel
        unsigned char r = d_rgb[rgbOffset];
        unsigned char g = d_rgb[rgbOffset + 1];
        unsigned char b = d_rgb[rgbOffset + 2];
        d_gray[grayOffset] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

int main() {
    const char* imagePath = "/content/Prova1.png"; // Path of the input image
    int width, height, channels;
    unsigned char *rgb = stbi_load(imagePath, &width, &height, &channels, 3);
    if (!rgb) {
        printf("Error loading image %s\n", imagePath);
        return 1;
    }
    printf("Image loaded: %dx%d with %d channels\n", width, height, channels);

    int imageSize = width * height;
    int rgbSize = imageSize * 3;
    unsigned char *h_gray = (unsigned char *)malloc(imageSize);
    unsigned char *cpu_gray = (unsigned char *)malloc(imageSize);
    rgbToGrayCPU(rgb, cpu_gray, width, height);

    unsigned char *d_rgb, *d_gray, *d_flipped;
    CHECK(cudaMalloc((void **)&d_rgb, rgbSize));
    CHECK(cudaMalloc((void **)&d_gray, imageSize));
    CHECK(cudaMalloc((void **)&d_flipped, rgbSize));

    CHECK(cudaMemcpy(d_rgb, rgb, rgbSize, cudaMemcpyHostToDevice));

    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    rgbToGrayGPU<<<grid, block>>>(d_rgb, d_gray, width, height);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(h_gray, d_gray, imageSize, cudaMemcpyDeviceToHost));

    bool match = true;
    for (int i = 0; i < imageSize; i++) {
        if (abs(cpu_gray[i] - h_gray[i]) > 1) {
            match = false;
            printf("Mismatch at pixel %d: CPU %d, GPU %d\n", i, cpu_gray[i], h_gray[i]);
            break;
        }
    }
    if (match) printf("CPU and GPU results match.\n");

    // Save the grayscale image
    stbi_write_png("output_gray.png", width, height, 1, h_gray, width);

    // Create flipped image (horizontal flip)
    unsigned char *h_flipped = (unsigned char *)malloc(rgbSize);
    cudaImageFlip<<<grid, block>>>(d_rgb, d_flipped, width, height, 3, true); // true for horizontal flip
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_flipped, d_flipped, rgbSize, cudaMemcpyDeviceToHost));

    // Save the flipped image
    stbi_write_png("output_flipped.png", width, height, 3, h_flipped, width * 3);

    // Free memory
    stbi_image_free(rgb);
    free(h_gray);
    free(cpu_gray);
    free(h_flipped);
    CHECK(cudaFree(d_rgb));
    CHECK(cudaFree(d_gray));
    CHECK(cudaFree(d_flipped));
    CHECK(cudaDeviceReset());

    return 0;
}
