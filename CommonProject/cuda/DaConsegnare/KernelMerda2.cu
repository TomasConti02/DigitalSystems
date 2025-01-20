//nvcc -Xptxas -O0 -arch=sm_60 your_kernel.cu -o your_kernel
//nvcc -O0 -Xptxas -O0 -arch=sm_60 -o your_kernel your_kernel.cu

#include <iostream>
#include <vector>
#include <cmath>
#include <sndfile.h>
#include <fftw3.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <chrono>
#include <cstring>
#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <chrono>
#define ALIGNMENT 32

#define SAMPLE_RATE 44100
#define BLOCK_SIZE 1024
#define ELEMENTS_PER_THREAD 1
__global__ void applyMultiBandGainKernel(float* real, float* imag, float* gains, int* bandLimits, int numSamples) {
    __shared__ float sharedReal[BLOCK_SIZE];
    __shared__ float sharedImag[BLOCK_SIZE];
    __shared__ float gainDaApplicare;
    const int tid = threadIdx.x;
    const int blockOffset = blockIdx.x * blockDim.x;
    const int globalOffset = blockOffset + tid;
    if (tid < BLOCK_SIZE) {
        sharedReal[tid] = real[tid];
        sharedImag[tid] = imag[tid] ;
    }
    __syncthreads();
    
    gainDaApplicare = gains[2];
    sharedReal[tid] = real[globalOffset] * gainDaApplicare;
    sharedImag[tid] = imag[globalOffset] * gainDaApplicare;

    if (globalOffset < bandLimits[0]) {
        gainDaApplicare = gains[0];  // Banda bassa
        sharedReal[tid] = real[globalOffset] * gainDaApplicare;
        sharedImag[tid] = imag[globalOffset] * gainDaApplicare;
    } else if (globalOffset < bandLimits[1]) {
        gainDaApplicare = gains[1];  // Banda media
        sharedReal[tid] = real[globalOffset] * gainDaApplicare;
        sharedImag[tid] = imag[globalOffset] * gainDaApplicare;
    }
    __syncthreads();

    if (globalOffset < numSamples) {
        // Scrittura dei risultati nella memoria globale
        real[globalOffset] = sharedReal[tid];
        imag[globalOffset] = sharedImag[tid];
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

    float* d_gains;
    int* d_bandLimits;
    cudaMalloc(&d_gains, sizeof(float) * 3);
    cudaMalloc(&d_bandLimits, sizeof(int) * 2);
    cudaMemcpy(d_gains, gains, sizeof(float) * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bandLimits, bandLimits, sizeof(int) * 2, cudaMemcpyHostToDevice);

    float *d_real, *d_imag;
    
    // Allocazione memoria sulla GPU
    cudaMalloc(&d_real, numSamples * sizeof(float));
    cudaMalloc(&d_imag, numSamples * sizeof(float));
    
    // Copia dei dati dalla CPU alla GPU
    cudaMemcpy(d_real, real, numSamples * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_imag, imag, numSamples * sizeof(float), cudaMemcpyHostToDevice);
    
    // Configurazione del lancio del kernel
    const int threadsPerBlock = BLOCK_SIZE;
    const int numBlocks = (numSamples + threadsPerBlock - 1) / threadsPerBlock;
    
    // Esecuzione del kernel
    auto start = std::chrono::high_resolution_clock::now();
    applyMultiBandGainKernel<<<numBlocks, threadsPerBlock>>>(d_real, d_imag, d_gains, d_bandLimits, numSamples);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    std::cout << "CUDA execution time: " 
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() 
              << " µs\n";
    
    // Copia dei risultati dalla GPU alla CPU
    cudaMemcpy(real, d_real, numSamples * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(imag, d_imag, numSamples * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Liberazione della memoria
    cudaFree(d_gains);
    cudaFree(d_bandLimits);
    cudaFree(d_real);
    cudaFree(d_imag);
}

int main(int argc, char* argv[]) {
    const char* inputFile = "/content/drive/MyDrive/Colab Notebooks/fullSong1.wav";
    const char* outputFile = "/content/drive/MyDrive/Colab Notebooks/fullSong5264.wav";
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
