#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h> // For fabs() function

#define CHECK(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
} \

// Function to measure time in seconds
double cpuSecond() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

// Sequential sum of two matrices on the host
void sumMatrixOnHost(float *MatA, float *MatB, float *MatC, int W, int H) {
    // Iterate through rows
    for (int i = 0; i < H; i++) { 
        // Iterate through columns
        for (int j = 0; j < W; j++) { 
            int idx = i * W + j; // Calculate linear index
            MatC[idx] = MatA[idx] + MatB[idx]; // Sum corresponding elements
        }
    }
}

// Kernel to sum matrices on the GPU
__global__ void sumMatrixOnGPU2D(float *MatA, float *MatB, float *MatC, int W, int H) {
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x; // Compute thread index for width
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y; // Compute thread index for height
    if (ix < W && iy < H) { // Ensure valid thread range
        unsigned int idx = iy * W + ix; // Compute global index of the thread
        MatC[idx] = MatA[idx] + MatB[idx]; // Sum the elements
    }
}

int main(int argc, char **argv) {
    // Initialize the device (GPU nvidia)
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));
    // Define matrix size (16384 x 16384) one element of the matrix will be execute by a GPU thread
    int W = 1 << 14; // Width of the matrix
    int H = 1 << 14; // Height of the matrix
    int size = W * H; // Total number of elements in the matrix
    int nBytes = size * sizeof(float); // Total memory size in bytes
    printf("Matrix size: W %d H %d\n", W, H);

    // Allocate memory for matrices on the HOST
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    // Initialize host matrices
    for (int i = 0; i < size; i++) {
        h_A[i] = 1.0f; // Initialize matrix A with 1.0f
        h_B[i] = 2.0f; // Initialize matrix B with 2.0f
    }

    // Allocate memory for matrices on the DEVICE
    float *d_MatA, *d_MatB, *d_MatC;
    CHECK(cudaMalloc((void **)&d_MatA, nBytes));
    CHECK(cudaMalloc((void **)&d_MatB, nBytes));
    CHECK(cudaMalloc((void **)&d_MatC, nBytes));

    // COPY matrices h_A and h_B HOST TO DEVICE 
    CHECK(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice));

    // Define block and grid size for kernel execution
    int block_dimx = 32;
    int block_dimy = 32;
    dim3 block(block_dimx, block_dimy); // Block size (32x32 threads per block)
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y); // Grid size based on matrix dimensions

    // Execute the GPU kernel
    double iStart = cpuSecond();
    sumMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, W, H);
    CHECK(cudaDeviceSynchronize()); // Ensure kernel execution is complete
    double gpuTime = cpuSecond() - iStart;
    printf("GPU Execution time: %f sec\n", gpuTime);

    // Copy result from device to host
    CHECK(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost));

    // Perform sequential sum on the host
    iStart = cpuSecond();
    sumMatrixOnHost(h_A, h_B, hostRef, W, H);
    double cpuTime = cpuSecond() - iStart;
    printf("CPU Execution time: %f sec\n", cpuTime);

    // Verify results between host and device
    bool match = true;
    for (int i = 0; i < size; i++) {
        if (fabs(hostRef[i] - gpuRef[i]) > 1e-5) { // Use fabs() for floating point comparison
            match = false;
            printf("Mismatch at index %d: host %f vs gpu %f\n", i, hostRef[i], gpuRef[i]);
            break;
        }
    }
    if (match) {
        printf("Results match!\n");
    } else {
        printf("Results do not match.\n");
    }

    // Free memory on device and host
    CHECK(cudaFree(d_MatA));
    CHECK(cudaFree(d_MatB));
    CHECK(cudaFree(d_MatC));
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    return 0;
}
