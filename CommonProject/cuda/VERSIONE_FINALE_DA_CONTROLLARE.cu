#include <iostream>
#include <vector>
#include <cmath>
#include <sndfile.h>
#include <fftw3.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <chrono>
#include <cstring>

#define SAMPLE_RATE 44100
#define BLOCK_SIZE 256
#define ELEMENTS_PER_THREAD 4
#define SHAREDSIZE ELEMENTS_PER_THREAD*BLOCK_SIZE

// Group all constant memory variables together
__constant__ float d_gains[3];  // [lowGain, midGain, highGain]
__constant__ int d_bandLimits[2];  // [lowEnd, midEnd]
__constant__ int d_warpSize;  // CUDA warp size
__global__ void applyMultiBandGainKernel(float* __restrict__ real, float* __restrict__ imag, const int numSamples) {
    __shared__ float sharedReal[SHAREDSIZE];
    __shared__ float sharedImag[SHAREDSIZE];
    
   
    // Thread indexing
    const int tid = threadIdx.x;
    const int warpId = tid / d_warpSize; //in quale warp si trova il thread
    const int laneId = tid % d_warpSize; //la posizione del thread dentro il warp identificato dal warpId
    const int baseIdx = blockIdx.x * (blockDim.x * ELEMENTS_PER_THREAD);

    // Load data into shared memory with coalesced access
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        const int globalIdx = baseIdx + laneId + (i * d_warpSize) + (warpId * d_warpSize * ELEMENTS_PER_THREAD);
        const int sharedIdx = tid + (i * blockDim.x);
        
        if (globalIdx < numSamples) {
            sharedReal[sharedIdx] = real[globalIdx];
            sharedImag[sharedIdx] = imag[globalIdx];
        }
    }
    
    __syncthreads();
    
    // Process and write back with coalesced access
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        const int globalIdx = baseIdx + laneId + (i * d_warpSize) + (warpId * d_warpSize * ELEMENTS_PER_THREAD);
        const int sharedIdx = tid + (i * blockDim.x);
        
        if (globalIdx < numSamples) {
            // Calculate gain only once per element
            float gain = (globalIdx < d_bandLimits[0]) ? d_gains[0] : 
                        (globalIdx < d_bandLimits[1]) ? d_gains[1] : d_gains[2];
            
            // Write back to global memory
            real[globalIdx] = sharedReal[sharedIdx] * gain;
            imag[globalIdx] = sharedImag[sharedIdx] * gain;
        }
    }
}
/*
__global__ void applyMultiBandGainKernel(float* __restrict__ real, float* __restrict__ imag, const int numSamples) {
    __shared__ float sharedReal[SHAREDSIZE];
    __shared__ float sharedImag[SHAREDSIZE];
    
    // Use d_warpSize from constant memory
    const int tid = threadIdx.x;
    const int warpId = tid / d_warpSize;
    const int laneId = tid % d_warpSize;
    
    const int blockOffset = blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD;
    
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        const int globalIdx = blockOffset + laneId + (i * d_warpSize) + (warpId * d_warpSize * ELEMENTS_PER_THREAD);
        const int sharedIdx = tid + i * blockDim.x;
        
        if (globalIdx < numSamples) {
            sharedReal[sharedIdx] = real[globalIdx];
            sharedImag[sharedIdx] = imag[globalIdx];
        }
    }
    
    __syncthreads();
    
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        const int globalIdx = blockOffset + laneId + (i * d_warpSize) + (warpId * d_warpSize * ELEMENTS_PER_THREAD);
        const int sharedIdx = tid + i * blockDim.x;
        
        if (globalIdx < numSamples) {
            float gain = (globalIdx < d_bandLimits[0]) ? d_gains[0] : 
                        (globalIdx < d_bandLimits[1]) ? d_gains[1] : d_gains[2];
            
            real[globalIdx] = sharedReal[sharedIdx] * gain;
            imag[globalIdx] = sharedImag[sharedIdx] * gain;
        }
    }
}

Analizziamo gli accessi alla memoria globale per ogni thread richiesto. Userò i valori BLOCK_SIZE=256 e ELEMENTS_PER_THREAD=4.

Thread 0 del primo blocco (blockIdx.x = 0):


tid = 0
warpId = 0/32 = 0
laneId = 0%32 = 0
baseIdx = 0 * (256 * 4) = 0

I suoi globalIdx saranno:

i=0: globalIdx = 0 + 0 + (0 * 32) + (0 * 32 * 4) = 0
i=1: globalIdx = 0 + 0 + (1 * 32) + (0 * 32 * 4) = 32
i=2: globalIdx = 0 + 0 + (2 * 32) + (0 * 32 * 4) = 64
i=3: globalIdx = 0 + 0 + (3 * 32) + (0 * 32 * 4) = 96


Thread 32 del primo blocco (blockIdx.x = 0):


tid = 32
warpId = 32/32 = 1
laneId = 32%32 = 0
baseIdx = 0 * (256 * 4) = 0

I suoi globalIdx saranno:

i=0: globalIdx = 0 + 0 + (0 * 32) + (1 * 32 * 4) = 128
i=1: globalIdx = 0 + 0 + (1 * 32) + (1 * 32 * 4) = 160
i=2: globalIdx = 0 + 0 + (2 * 32) + (1 * 32 * 4) = 192
i=3: globalIdx = 0 + 0 + (3 * 32) + (1 * 32 * 4) = 224


Thread 0 del secondo blocco (blockIdx.x = 1):


tid = 0
warpId = 0/32 = 0
laneId = 0%32 = 0
baseIdx = 1 * (256 * 4) = 1024

I suoi globalIdx saranno:

i=0: globalIdx = 1024 + 0 + (0 * 32) + (0 * 32 * 4) = 1024
i=1: globalIdx = 1024 + 0 + (1 * 32) + (0 * 32 * 4) = 1056
i=2: globalIdx = 1024 + 0 + (2 * 32) + (0 * 32 * 4) = 1088
i=3: globalIdx = 1024 + 0 + (3 * 32) + (0 * 32 * 4) = 1120

Quindi:

Il thread 0 del blocco 0 accede agli indici: [0, 32, 64, 96]
Il thread 32 del blocco 0 accede agli indici: [128, 160, 192, 224]
Il thread 0 del blocco 1 accede agli indici: [1024, 1056, 1088, 1120]

Questi indici rappresentano gli accessi sia in lettura (array real e imag) che in scrittura.
Per i=0:
Thread 0:  globalIdx = 0 + 0 + 0 = 0
Thread 1:  globalIdx = 0 + 1 + 0 = 1
Thread 2:  globalIdx = 0 + 2 + 0 = 2
...
Thread 31: globalIdx = 0 + 31 + 0 = 31

Per i=1:
Thread 0:  globalIdx = 0 + 0 + 32 = 32
Thread 1:  globalIdx = 0 + 1 + 32 = 33
Thread 2:  globalIdx = 0 + 2 + 32 = 34
...
Thread 31: globalIdx = 0 + 31 + 32 = 63
Per i=0:
Thread 32: globalIdx = 0 + 0 + 128 = 128
Thread 33: globalIdx = 0 + 1 + 128 = 129
Thread 34: globalIdx = 0 + 2 + 128 = 130
...
Thread 63: globalIdx = 0 + 31 + 128 = 159
*/
void applyCudaEqualizer(float* real, float* imag, int numSamples, int sampleRate) {
    int bandLimits[2];
    bandLimits[0] = static_cast<int>(300.0f / (static_cast<float>(sampleRate) / numSamples));
    bandLimits[1] = static_cast<int>(3000.0f / (static_cast<float>(sampleRate) / numSamples));
    
    float gains[3] = {
        std::pow(10.0f, -60.0f / 20.0f),
        std::pow(10.0f, 2.0f / 20.0f),
        std::pow(10.0f, -3.0f / 20.0f)
    };
    
    // Initialize warpSize value
    int warpSize = 32;
    
    // Copy all constant values to device memory
    cudaMemcpyToSymbol(d_gains, gains, sizeof(float) * 3);
    cudaMemcpyToSymbol(d_bandLimits, bandLimits, sizeof(int) * 2);
    cudaMemcpyToSymbol(d_warpSize, &warpSize, sizeof(int));
    
    float *d_real, *d_imag;
    cudaMalloc(&d_real, numSamples * sizeof(float));
    cudaMalloc(&d_imag, numSamples * sizeof(float));
    
    cudaMemcpy(d_real, real, numSamples * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_imag, imag, numSamples * sizeof(float), cudaMemcpyHostToDevice);
    
    const int threadsPerBlock = BLOCK_SIZE;
    const int numBlocks = (numSamples + (threadsPerBlock * ELEMENTS_PER_THREAD) - 1) / (threadsPerBlock * ELEMENTS_PER_THREAD);
    
    auto start = std::chrono::high_resolution_clock::now();
    applyMultiBandGainKernel<<<numBlocks, threadsPerBlock>>>(d_real, d_imag, numSamples);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    cudaMemcpy(real, d_real, numSamples * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(imag, d_imag, numSamples * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_real);
    cudaFree(d_imag);
}

int main(int argc, char* argv[]) {
    const char* inputFile = "/content/drive/MyDrive/Colab Notebooks/fullSong1.wav";
    const char* outputFile = "/content/drive/MyDrive/Colab Notebooks/fullSongsong.wav";
    SF_INFO sfInfo{};
    SNDFILE* inFile = sf_open(inputFile, SFM_READ, &sfInfo);
    
    if (!inFile) {
        std::cerr << "[ERR] Reading input file" << std::endl;
        return 1;
    }
    
    int numSamples = sfInfo.frames * sfInfo.channels;
    
    // Allocate real and imag arrays in pinned memory
    float *real, *imag;
    cudaError_t err = cudaMallocHost(&real, sizeof(float) * numSamples);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate pinned memory for real array: " 
                  << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    err = cudaMallocHost(&imag, sizeof(float) * numSamples);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate pinned memory for imag array: " 
                  << cudaGetErrorString(err) << std::endl;
        cudaFreeHost(real);  // Clean up previously allocated memory
        return 1;
    }
    
    std::vector<short> buffer(numSamples);
    sf_read_short(inFile, buffer.data(), numSamples);
    sf_close(inFile);
    
    for (int i = 0; i < numSamples; ++i) {
        real[i] = static_cast<float>(buffer[i]);
    }
    std::memset(imag, 0, sizeof(float) * numSamples);
    
    fftwf_complex* fftData = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * numSamples);
    fftwf_plan forwardPlan = fftwf_plan_dft_r2c_1d(numSamples, real, fftData, FFTW_ESTIMATE);
    fftwf_plan inversePlan = fftwf_plan_dft_c2r_1d(numSamples, fftData, real, FFTW_ESTIMATE);
    
    fftwf_execute(forwardPlan);
    
    for (int i = 0; i < numSamples / 2 + 1; ++i) {
        real[i] = fftData[i][0];
        imag[i] = fftData[i][1];
    }
    
    applyCudaEqualizer(real, imag, numSamples / 2, SAMPLE_RATE);
    
    for (int i = 0; i < numSamples / 2 + 1; ++i) {
        fftData[i][0] = real[i];
        fftData[i][1] = imag[i];
    }
    
    fftwf_execute(inversePlan);
    
    float normalFactor = 1.0f / numSamples;
    for (int i = 0; i < numSamples; ++i) {
        buffer[i] = static_cast<short>(std::round(real[i] * normalFactor));
    }
    
    SNDFILE* outFile = sf_open(outputFile, SFM_WRITE, &sfInfo);
    if (!outFile) {
        std::cerr << "[ERR] Writing output file" << std::endl;
        cudaFreeHost(real);
        cudaFreeHost(imag);
        return 1;
    }
    
    sf_write_short(outFile, buffer.data(), numSamples);
    sf_close(outFile);
    
    fftwf_destroy_plan(forwardPlan);
    fftwf_destroy_plan(inversePlan);
    fftwf_free(fftData);
    
    cudaFreeHost(real);
    cudaFreeHost(imag);
    
    return 0;
}
