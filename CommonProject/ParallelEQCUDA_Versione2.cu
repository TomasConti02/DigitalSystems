#include <iostream>
#include <vector>
#include <cmath>
#include <sndfile.h>
#include <fftw3.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cstring>

#define SAMPLE_RATE 44100
#define BLOCK_SIZE 256

// Costanti in memoria costante
__constant__ float d_gains[3];  // [lowGain, midGain, highGain]
__constant__ int d_bandLimits[2];  // [lowEnd, midEnd]
//Testare il kernel senza l'uso di sincronizzazioni + testare il kernel variando BLOCK_SIZE
__global__ void applyMultiBandGainKernel(float* real, float* imag, int numSamples) {
    __shared__ float shared_real[BLOCK_SIZE];
    __shared__ float shared_imag[BLOCK_SIZE];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Carica i dati nella memoria condivisa
    if (idx < numSamples) {
        shared_real[threadIdx.x] = real[idx];
        shared_imag[threadIdx.x] = imag[idx];
    }
    __syncthreads();
    
    if (idx < numSamples) {
        float gain;
        if (idx < d_bandLimits[0]) {
            gain = d_gains[0];  // Low gain
        } else if (idx < d_bandLimits[1]) {
            gain = d_gains[1];  // Mid gain
        } else {
            gain = d_gains[2];  // High gain
        }
        
        shared_real[threadIdx.x] *= gain;
        shared_imag[threadIdx.x] *= gain;
    }
    __syncthreads();
    
    // Scrivi i risultati in memoria globale
    if (idx < numSamples) {
        real[idx] = shared_real[threadIdx.x];
        imag[idx] = shared_imag[threadIdx.x];
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
    size_t pitch;
    
    // Allocazione allineata della memoria sulla GPU
    cudaMallocPitch(&d_real, &pitch, numSamples * sizeof(float), 1);
    cudaMallocPitch(&d_imag, &pitch, numSamples * sizeof(float), 1);
    
    // Copia dei dati dalla CPU alla GPU
    cudaMemcpy2D(d_real, pitch, real, numSamples * sizeof(float),
                 numSamples * sizeof(float), 1, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_imag, pitch, imag, numSamples * sizeof(float),
                 numSamples * sizeof(float), 1, cudaMemcpyHostToDevice);
    
    // Ottimizzazione della dimensione della griglia
    int numSM;
    cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, 0);
    int numBlocks = (numSamples + BLOCK_SIZE - 1) / BLOCK_SIZE;
    numBlocks = (numBlocks + numSM - 1) / numSM * numSM; // Arrotonda al multiplo superiore di SM
    
    dim3 gridDim(numBlocks);
    dim3 blockDim(BLOCK_SIZE);
    
    // Esecuzione del kernel ottimizzato
    applyMultiBandGainKernel<<<gridDim, blockDim>>>(d_real, d_imag, numSamples);
    
    // Copia dei risultati dalla GPU alla CPU
    cudaMemcpy2D(real, numSamples * sizeof(float), d_real, pitch,
                 numSamples * sizeof(float), 1, cudaMemcpyDeviceToHost);
    cudaMemcpy2D(imag, numSamples * sizeof(float), d_imag, pitch,
                 numSamples * sizeof(float), 1, cudaMemcpyDeviceToHost);
    
    // Liberazione della memoria
    cudaFree(d_real);
    cudaFree(d_imag);
    
    // Stampa delle configurazioni
    printf("=== Equalizer Configuration ===\n");
    printf("Sample rate: %d Hz\n", sampleRate);
    printf("Number of samples: %d\n", numSamples);
    printf("Low band range: 0 to %d\n", bandLimits[0]);
    printf("Mid band range: %d to %d\n", bandLimits[0], bandLimits[1]);
    printf("High band range: %d to %d\n", bandLimits[1], numSamples);
    printf("Low band gain: %.2f\n", gains[0]);
    printf("Mid band gain: %.2f\n", gains[1]);
    printf("High band gain: %.2f\n", gains[2]);
    
    printf("\n=== Kernel Configuration ===\n");
    printf("Grid size: %d blocks\n", gridDim.x);
    printf("Block size: %d threads per block\n", blockDim.x);
    printf("Total threads: %d\n", gridDim.x * blockDim.x);
    printf("Number of SMs: %d\n", numSM);
}

int main(int argc, char* argv[]) {
    const char* inputFile = "/content/drive/MyDrive/Colab Notebooks/fullSong1.wav";
    const char* outputFile = "/content/drive/MyDrive/Colab Notebooks/fullSong7.wav";
    SF_INFO sfInfo{};
    SNDFILE* inFile = sf_open(inputFile, SFM_READ, &sfInfo);
    
    if (!inFile) {
        std::cerr << "[ERR] Reading input file" << std::endl;
        return 1;
    }
    
    int numSamples = sfInfo.frames * sfInfo.channels;
    float* real = (float*)std::aligned_alloc(16, sizeof(float) * numSamples);
    float* imag = (float*)std::aligned_alloc(16, sizeof(float) * numSamples);
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
