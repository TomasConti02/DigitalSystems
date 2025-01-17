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
#define BLOCK_SIZE 64
#define ELEMENTS_PER_THREAD 2
#define SHAREDSIZE ELEMENTS_PER_THREAD*BLOCK_SIZE
__constant__ float d_gains[3];  // [lowGain, midGain, highGain]
__constant__ int d_bandLimits[2];  // [lowEnd, midEnd]
/*
__global__ void applyMultiBandGainKernel(float* __restrict__ real, float* __restrict__ imag, const int numSamples) {
    __shared__ float sharedReal[SHAREDSIZE];
    __shared__ float sharedImag[SHAREDSIZE];
    
    // Calcola gli indici per accessi coalescenti
    const int tid = threadIdx.x;
    const int warpSize = 32;
    const int warpId = tid / warpSize;
    const int laneId = tid % warpSize;
    
    // Calcola l'offset base per il blocco
    const int blockOffset = blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD;
    
    // Per ogni elemento da processare per thread
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        // Calcola l'indice globale con accesso coalescente
        const int globalIdx = blockOffset + laneId + (i * warpSize) + (warpId * warpSize * ELEMENTS_PER_THREAD);
        const int sharedIdx = tid + i * blockDim.x;
        
        if (globalIdx < numSamples) {
            // Carica i dati in shared memory con accessi coalescenti
            sharedReal[sharedIdx] = real[globalIdx];
            sharedImag[sharedIdx] = imag[globalIdx];
        }
    }
    
    __syncthreads();
    
    // Ricalcola gli indici per il processing
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        const int globalIdx = blockOffset + laneId + (i * warpSize) + (warpId * warpSize * ELEMENTS_PER_THREAD);
        const int sharedIdx = tid + i * blockDim.x;
        
        if (globalIdx < numSamples) {
            // Determina il gain in base alla frequenza
            float gain = (globalIdx < d_bandLimits[0]) ? d_gains[0] : 
                        (globalIdx < d_bandLimits[1]) ? d_gains[1] : d_gains[2];
            
            // Scrivi il risultato con accessi coalescenti
            real[globalIdx] = sharedReal[sharedIdx] * gain;
            imag[globalIdx] = sharedImag[sharedIdx] * gain;
        }
    }
}*/

__global__ void applyMultiBandGainKernel(float* __restrict__ real, float* __restrict__ imag, const int numSamples) {
    __shared__ float sharedReal[SHAREDSIZE];
    __shared__ float sharedImag[SHAREDSIZE];
    
    // Calcola gli indici base per accessi coalescenti
    const int tid = threadIdx.x;
    //const int totalThreads = blockDim.x * gridDim.x;
    const int baseIdx = blockIdx.x * blockDim.x + tid;
    
    // Carica i dati in shared memory con accessi coalescenti
    const int Nuovo=baseIdx*ELEMENTS_PER_THREAD;
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        const int globalIdx = Nuovo + i;
        const int sharedIdx = tid + i * blockDim.x;
        
        if (baseIdx < numSamples) {
            // Accesso coalescente alla memoria globale
            sharedReal[sharedIdx] = real[globalIdx];
            sharedImag[sharedIdx] = imag[globalIdx];
        }
    }
    __syncthreads();
    float gain = (Nuovo < d_bandLimits[0]) ? d_gains[0] : 
                        (Nuovo < d_bandLimits[1]) ? d_gains[1] : d_gains[2];
    // Processa i dati e scrivi il risultato
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        const int globalIdx = Nuovo + i ;
        const int sharedIdx = tid + i * blockDim.x;
        
        if (baseIdx < numSamples) {
            // Calcola il gain una sola volta per ogni elemento
            // Accesso coalescente alla memoria globale
            real[globalIdx] = sharedReal[sharedIdx] * gain;
            imag[globalIdx] = sharedImag[sharedIdx] * gain;
        }
    }
}

void applyCudaEqualizer(float* real, float* imag, int numSamples, int sampleRate) {
    int bandLimits[2];
    bandLimits[0] = static_cast<int>(300.0f / (static_cast<float>(sampleRate) / numSamples));
    bandLimits[1] = static_cast<int>(3000.0f / (static_cast<float>(sampleRate) / numSamples));
    
    float gains[3] = {
        std::pow(10.0f, -60.0f / 20.0f),
        std::pow(10.0f, 2.0f / 20.0f),
        std::pow(10.0f, -3.0f / 20.0f)
    };
    
    cudaMemcpyToSymbol(d_gains, gains, sizeof(float) * 3);
    cudaMemcpyToSymbol(d_bandLimits, bandLimits, sizeof(int) * 2);
    
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
    const char* outputFile = "/content/drive/MyDrive/Colab Notebooks/fullSongaaaaaaaaaaaaaaaaaaa.wav";
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
