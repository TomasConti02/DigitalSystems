//sudo apt update
//!sudo apt install libfftw3-dev
//!nvcc -o CudaEQ prova.cu -lsndfile -lfftw3f -lcufft
//!ncu --kernel-name applyMultiBandGainKernelOptimized ./CudaEQ
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
#define ELEMENTS_PER_THREAD 8
#define ALIGNMENT 32  // Allineamento della memoria a 32 byte

// Costanti in memoria costante
__constant__ float d_gains[3];  // [lowGain, midGain, highGain]
__constant__ int d_bandLimits[2];  // [lowEnd, midEnd]

// Kernel ottimizzato
__global__ void applyMultiBandGainKernelOptimized(float4* __restrict__ real, float4* __restrict__ imag, const int numSamples) {
    __shared__ float4 sharedReal[BLOCK_SIZE];
    __shared__ float4 sharedImag[BLOCK_SIZE];

    const int tid = threadIdx.x;
    const int blockOffset = blockIdx.x * blockDim.x;
    const int globalIdx = blockOffset + tid;

    // Caricamento coalescente usando float4
    if (globalIdx * 4 < numSamples) {
        sharedReal[tid] = real[globalIdx];
        sharedImag[tid] = imag[globalIdx];
    }
    __syncthreads();

    // Processamento usando float4
    if (globalIdx * 4 < numSamples) {
        float4 rData = sharedReal[tid];
        float4 iData = sharedImag[tid];
        
        // Applica i guadagni a ogni componente
        int baseIdx = globalIdx * 4;
        float4 gains;
        gains.x = (baseIdx < d_bandLimits[0]) ? d_gains[0] : 
                 (baseIdx < d_bandLimits[1]) ? d_gains[1] : d_gains[2];
        gains.y = ((baseIdx + 1) < d_bandLimits[0]) ? d_gains[0] : 
                 ((baseIdx + 1) < d_bandLimits[1]) ? d_gains[1] : d_gains[2];
        gains.z = ((baseIdx + 2) < d_bandLimits[0]) ? d_gains[0] : 
                 ((baseIdx + 2) < d_bandLimits[1]) ? d_gains[1] : d_gains[2];
        gains.w = ((baseIdx + 3) < d_bandLimits[0]) ? d_gains[0] : 
                 ((baseIdx + 3) < d_bandLimits[1]) ? d_gains[1] : d_gains[2];

        // Applica i guadagni
        rData.x *= gains.x; rData.y *= gains.y; rData.z *= gains.z; rData.w *= gains.w;
        iData.x *= gains.x; iData.y *= gains.y; iData.z *= gains.z; iData.w *= gains.w;

        // Scrittura coalescente
        real[globalIdx] = rData;
        imag[globalIdx] = iData;
    }
}

void applyCudaEqualizer(float* real, float* imag, int numSamples, int sampleRate) {
    int bandLimits[2];
    bandLimits[0] = static_cast<int>(300.0f / (static_cast<float>(sampleRate) / numSamples));  // lowEnd
    bandLimits[1] = static_cast<int>(3000.0f / (static_cast<float>(sampleRate) / numSamples)); // midEnd
    float gains[3] = {
    std::pow(10.0f, -60.0f / 20.0f),  // LOW_GAIN
    std::pow(10.0f, 2.0f / 20.0f),    // MID_GAIN
    std::pow(10.0f, -3.0f / 20.0f)    // HIGH_GAIN  
    };
    cudaMemcpyToSymbol(d_gains, gains, sizeof(float) * 3);
    cudaMemcpyToSymbol(d_bandLimits, bandLimits, sizeof(int) * 2);
    // Calcolo della memoria richiesta per float4
    int numFloat4 = (numSamples + 3) / 4;
    size_t sizeRealImag = numFloat4 * sizeof(float4);

    // Alloca memoria sul dispositivo
    float4* d_real;
    float4* d_imag;
    cudaMalloc((void**)&d_real, sizeRealImag);
    cudaMalloc((void**)&d_imag, sizeRealImag);

    // Copia i dati reali e immaginari dal host al dispositivo
    cudaMemcpy(d_real, real, numSamples * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_imag, imag, numSamples * sizeof(float), cudaMemcpyHostToDevice);

    // Copia costanti per i limiti delle bande e i guadagni
    cudaMemcpyToSymbol(d_bandLimits, bandLimits, 3 * sizeof(int));
    cudaMemcpyToSymbol(d_gains, gains, 3 * sizeof(float));

    // Configurazione del kernel
    const int threadsPerBlock = BLOCK_SIZE;
    const int numBlocks = (numFloat4 + threadsPerBlock - 1) / threadsPerBlock;

    // Lancio del kernel
    applyMultiBandGainKernelOptimized<<<numBlocks, threadsPerBlock>>>(d_real, d_imag, numSamples);
    cudaDeviceSynchronize();

    // Copia i dati elaborati dal dispositivo al host
    cudaMemcpy(real, d_real, numSamples * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(imag, d_imag, numSamples * sizeof(float), cudaMemcpyDeviceToHost);

    // Libera memoria sul dispositivo
    cudaFree(d_real);
    cudaFree(d_imag);
}

int main(int argc, char* argv[]) {
    const char* inputFile = "/content/drive/MyDrive/Colab Notebooks/fullSong1.wav";
    const char* outputFile = "/content/drive/MyDrive/Colab Notebooks/fullSong200.wav";
    SF_INFO sfInfo{};
    SNDFILE* inFile = sf_open(inputFile, SFM_READ, &sfInfo);
    
    if (!inFile) {
        std::cerr << "[ERR] Reading input file" << std::endl;
        return 1;
    }
    
    int numSamples = sfInfo.frames * sfInfo.channels;
    float* real = (float*)std::aligned_alloc(ALIGNMENT, sizeof(float) * numSamples);
    float* imag = (float*)std::aligned_alloc(ALIGNMENT, sizeof(float) * numSamples);
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
        return 1;
    }
    
    sf_write_short(outFile, buffer.data(), numSamples);
    sf_close(outFile);
    
    fftwf_destroy_plan(forwardPlan);
    fftwf_destroy_plan(inversePlan);
    fftwf_free(fftData);
    free(real);
    free(imag);
    
    return 0;
}
/*//sudo apt update
//!sudo apt install libfftw3-dev
//!nvcc -o CudaEQ prova.cu -lsndfile -lfftw3f -lcufft
//!ncu --kernel-name applyMultiBandGainKernelOptimized ./CudaEQ
size_t size = ((n * sizeof(float) + 511) / 512) * 512;
cudaMalloc(&d_data, size);*/
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
#define MEMORY_ALIGNMENT 512  // 32 thread × 4 float × 4 bytes

// Costanti in memoria costante
__constant__ float d_gains[3];  // [lowGain, midGain, highGain]
__constant__ int d_bandLimits[2];  // [lowEnd, midEnd]
size_t getAlignedSize(size_t size) {
    return ((size + MEMORY_ALIGNMENT - 1) / MEMORY_ALIGNMENT) * MEMORY_ALIGNMENT;
}
// Kernel ottimizzato
__global__ void applyMultiBandGainKernelOptimized(float* __restrict__ real, float* __restrict__ imag, const int numSamples) {
    __shared__ float sharedReal[BLOCK_SIZE * ELEMENTS_PER_THREAD];
    __shared__ float sharedImag[BLOCK_SIZE * ELEMENTS_PER_THREAD];

    const int tid = threadIdx.x;
    const int blockOffset = blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD;
    const int globalOffset = blockOffset + tid;

    // Caricamento nella memoria condivisa (con accesso coalescente)
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        const int idx = globalOffset + i * blockDim.x;
        if (idx < numSamples) {
            sharedReal[tid + i * blockDim.x] = real[idx];
            sharedImag[tid + i * blockDim.x] = imag[idx];
        }
    }
    __syncthreads();

    // Applicazione dei guadagni
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        const int idx = globalOffset + i * blockDim.x;
        if (idx < numSamples) {
            float gain = (idx < d_bandLimits[0]) * d_gains[0] +
                         (idx >= d_bandLimits[0] && idx < d_bandLimits[1]) * d_gains[1] +
                         (idx >= d_bandLimits[1]) * d_gains[2];
            sharedReal[tid + i * blockDim.x] *= gain;
            sharedImag[tid + i * blockDim.x] *= gain;
        }
    }
    //__syncthreads(); non necessariamente necessante 

    // Scrittura nella memoria globale
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        const int idx = globalOffset + i * blockDim.x;
        if (idx < numSamples) {
            real[idx] = sharedReal[tid + i * blockDim.x];
            imag[idx] = sharedImag[tid + i * blockDim.x];
        }
    }
}

void applyCudaEqualizer(float* real, float* imag, int numSamples, int sampleRate) {
    // Calcolo degli intervalli delle bande
    int bandLimits[2];
    bandLimits[0] = static_cast<int>(300.0f / (static_cast<float>(sampleRate) / numSamples));  // lowEnd
    bandLimits[1] = static_cast<int>(3000.0f / (static_cast<float>(sampleRate) / numSamples)); // midEnd
    
    // Calcolo dei guadagni
    float gains[3] = {
        std::pow(10.0f, -60.0f / 20.0f),  // LOW_GAIN
        std::pow(10.0f, 2.0f / 20.0f),    // MID_GAIN
        std::pow(10.0f, -3.0f / 20.0f)    // HIGH_GAIN
    };
    
    // Copia delle costanti in memoria costante
    cudaMemcpyToSymbol(d_gains, gains, sizeof(float) * 3);
    cudaMemcpyToSymbol(d_bandLimits, bandLimits, sizeof(int) * 2);
    
    float *d_real, *d_imag;
    size_t alignedSize = getAlignedSize(numSamples * sizeof(float));
    
    
    // Allocazione memoria sulla GPU con allineamento
    cudaMalloc(&d_real, alignedSize);
    cudaMalloc(&d_imag, alignedSize);
    
    // Copia dei dati dalla CPU alla GPU
    cudaMemcpy(d_real, real, numSamples * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_imag, imag, numSamples * sizeof(float), cudaMemcpyHostToDevice);
    
    // Configurazione del lancio del kernel
    const int threadsPerBlock = BLOCK_SIZE;
    const int numBlocks = (numSamples + (threadsPerBlock * ELEMENTS_PER_THREAD) - 1) / (threadsPerBlock * ELEMENTS_PER_THREAD);
    // Esecuzione del kernel
    auto start = std::chrono::high_resolution_clock::now();
    applyMultiBandGainKernelOptimized<<<numBlocks, threadsPerBlock>>>(d_real, d_imag, numSamples);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "CUDA execution time Optimized: " 
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() 
              << " µs\n";
    
    // Copia dei risultati dalla GPU alla CPU
    cudaMemcpy(real, d_real, numSamples * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(imag, d_imag, numSamples * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Liberazione della memoria
    cudaFree(d_real);
    cudaFree(d_imag);
    
    // Stampa delle configurazioni
    printf("\n=== Equalizer Configuration ===\n");
    printf("Sample rate: %d Hz\n", sampleRate);
    printf("Number of samples: %d\n", numSamples);
    printf("Low band range: 0 to %d\n", bandLimits[0]);
    printf("Mid band range: %d to %d\n", bandLimits[0], bandLimits[1]);
    printf("High band range: %d to %d\n", bandLimits[1], numSamples);
    printf("Low band gain: %.2f\n", gains[0]);
    printf("Mid band gain: %.2f\n", gains[1]);
    printf("High band gain: %.2f\n", gains[2]);
    
    printf("\n=== Kernel Configuration ===\n");
    printf("Grid size: %d blocks\n", numBlocks);
    printf("Block size: %d threads per block\n", BLOCK_SIZE);
    printf("Elements per thread: %d\n", ELEMENTS_PER_THREAD);
    printf("Total threads: %d\n", numBlocks * BLOCK_SIZE);
}

int main(int argc, char* argv[]) {
    const char* inputFile = "/content/drive/MyDrive/Colab Notebooks/fullSong1.wav";
    const char* outputFile = "/content/drive/MyDrive/Colab Notebooks/fullSong111.wav";
    SF_INFO sfInfo{};
    SNDFILE* inFile = sf_open(inputFile, SFM_READ, &sfInfo);
    
    if (!inFile) {
        std::cerr << "[ERR] Reading input file" << std::endl;
        return 1;
    }
    
    int numSamples = sfInfo.frames * sfInfo.channels;
    float* real = (float*)std::aligned_alloc(MEMORY_ALIGNMENT, sizeof(float) * numSamples);
    float* imag = (float*)std::aligned_alloc(MEMORY_ALIGNMENT, sizeof(float) * numSamples);
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
        return 1;
    }
    
    sf_write_short(outFile, buffer.data(), numSamples);
    sf_close(outFile);
    
    fftwf_destroy_plan(forwardPlan);
    fftwf_destroy_plan(inversePlan);
    fftwf_free(fftData);
    free(real);
    free(imag);
    
    return 0;
}*/
