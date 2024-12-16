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

//fanno parte della memoria costante
const float LOW_GAIN = -60.0f;
const float MID_GAIN = 2.0f;
const float HIGH_GAIN= -3.0f;
/*
=== Equalizer Configuration ===
Sample rate: 44100 Hz
Number of samples: 6526800
Low band range: 0 to 44400
Mid band range: 44400 to 444000
High band range: 444000 to 6526800
Low band gain: 0.00
Mid band gain: 1.26
High band gain: 0.71

=== Kernel Configuration ===
Low band kernel:
  Grid size: 174 blocks
  Block size: 256 threads per block
  Total threads: 44544
Mid band kernel:
  Grid size: 1561 blocks
  Block size: 256 threads per block
  Total threads: 399616
High band kernel:
  Grid size: 23761 blocks
  Block size: 256 threads per block
  Total threads: 6082816
*/
__global__ void applyGainKernel(float* real, float* imag, int startIdx, int endIdx, float gain) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + startIdx;
    if (idx < endIdx) {
        real[idx] *= gain;
        imag[idx] *= gain;
    }
}
void applyCudaEqualizer(float* real, float* imag, int numSamples, int sampleRate) {
    //TUTTI CALCOLI 1D
    float *d_real, *d_imag;

    // Calcolo degli intervalli delle bande
    int lowBandEnd = static_cast<int>(300.0f / (static_cast<float>(sampleRate) / numSamples));
    int midBandStart = lowBandEnd;
    int midBandEnd = static_cast<int>(3000.0f / (static_cast<float>(sampleRate) / numSamples));
    int highBandStart = midBandEnd;

    // Calcolo dei guadagni
    float lowGain = std::pow(10.0f, LOW_GAIN / 20.0f);
    float midGain = std::pow(10.0f, MID_GAIN / 20.0f);
    float highGain = std::pow(10.0f, HIGH_GAIN / 20.0f);
    float midGain_Device, lowGain_Device, highGain_Device;
    cudaMemcpyToSymbol(midGain_Device, &lowGain, sizeof(float));
    cudaMemcpyToSymbol(lowGain_Device, &midGain, sizeof(float));
    cudaMemcpyToSymbol(highGain_Device, &highGain, sizeof(float));
    // Allocazione della memoria sulla GPU
    cudaMalloc(&d_real, numSamples * sizeof(float));
    cudaMalloc(&d_imag, numSamples * sizeof(float));

    // Copia dei dati dalla CPU alla GPU
    cudaMemcpy(d_real, real, numSamples * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_imag, imag, numSamples * sizeof(float), cudaMemcpyHostToDevice);

    // Creazione di stream
    cudaStream_t streamLow, streamMid, streamHigh;
    cudaStreamCreate(&streamLow);
    cudaStreamCreate(&streamMid);
    cudaStreamCreate(&streamHigh);
    // Configurazione della griglia e dei blocchi
    dim3 blockDim(BLOCK_SIZE);
    // Kernel 1: Banda bassa (0 - lowBandEnd)
    int lowSize = lowBandEnd;
    dim3 gridDimLow((lowSize + BLOCK_SIZE - 1) / BLOCK_SIZE); //quanti blocchi da 256 thread mi servono per coprire lowsize dati e quindi anche un thread per dato
    //ho lowsize dati=lowsize thread suddivisi in gridDimLow blocchi che formano la mia griglia 1D
    applyGainKernel<<<gridDimLow, blockDim, 0, streamLow>>>(d_real, d_imag, 0, lowBandEnd, lowGain_Device);
    // Kernel 2: Banda media (midBandStart - midBandEnd)
    int midSize = midBandEnd - midBandStart;
    dim3 gridDimMid((midSize + BLOCK_SIZE - 1) / BLOCK_SIZE);
    //sintassi -> kernel<<<gridDim, blockDim, sharedMemBytes, stream>>>(...)
    //0: Quantità di memoria condivisa dinamica
    applyGainKernel<<<gridDimMid, blockDim, 0, streamMid>>>(d_real, d_imag, midBandStart, midBandEnd, midGain_Device);
    // Kernel 3: Banda alta (highBandStart - numSamples)
    int highSize = numSamples - highBandStart;
    dim3 gridDimHigh((highSize + BLOCK_SIZE - 1) / BLOCK_SIZE);
    applyGainKernel<<<gridDimHigh, blockDim, 0, streamHigh>>>(d_real, d_imag, highBandStart, numSamples, highGain_Device);
    // Sincronizzazione dei kernel (opzionale per assicurare il completamento)
    cudaStreamSynchronize(streamLow);
    cudaStreamSynchronize(streamMid);
    cudaStreamSynchronize(streamHigh);
    // Copia dei dati dalla GPU alla CPU
    cudaMemcpy(real, d_real, numSamples * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(imag, d_imag, numSamples * sizeof(float), cudaMemcpyDeviceToHost);
    // Liberazione della memoria GPU
    cudaFree(d_real);
    cudaFree(d_imag);
    // Distruzione degli stream
    cudaStreamDestroy(streamLow);
    cudaStreamDestroy(streamMid);
    cudaStreamDestroy(streamHigh);
    printf("=== Equalizer Configuration ===\n");
    printf("Sample rate: %d Hz\n", sampleRate);
    printf("Number of samples: %d\n", numSamples);
    printf("Low band range: 0 to %d\n", lowBandEnd);
    printf("Mid band range: %d to %d\n", midBandStart, midBandEnd);
    printf("High band range: %d to %d\n", highBandStart, numSamples);
    printf("Low band gain: %.2f\n", lowGain);
    printf("Mid band gain: %.2f\n", midGain);
    printf("High band gain: %.2f\n", highGain);
    printf("\n=== Kernel Configuration ===\n");
    // Informazioni sul primo kernel (banda bassa)
    printf("Low band kernel:\n");
    printf("  Grid size: %d blocks\n", gridDimLow.x);
    printf("  Block size: %d threads per block\n", blockDim.x);
    printf("  Total threads: %d\n", gridDimLow.x * blockDim.x);
    // Informazioni sul secondo kernel (banda media)
    printf("Mid band kernel:\n");
    printf("  Grid size: %d blocks\n", gridDimMid.x);
    printf("  Block size: %d threads per block\n", blockDim.x);
    printf("  Total threads: %d\n", gridDimMid.x * blockDim.x);
    // Informazioni sul terzo kernel (banda alta)
    printf("High band kernel:\n");
    printf("  Grid size: %d blocks\n", gridDimHigh.x);
    printf("  Block size: %d threads per block\n", blockDim.x);
    printf("  Total threads: %d\n", gridDimHigh.x * blockDim.x);
}

int main(int argc, char* argv[]) {
    const char* inputFile = "/content/drive/MyDrive/Colab Notebooks/fullSong1.wav";
    const char* outputFile = "/content/drive/MyDrive/Colab Notebooks/fullSong2.wav";
    SF_INFO sfInfo{};
    SNDFILE* inFile = sf_open(inputFile, SFM_READ, &sfInfo);

    if (!inFile) {
        std::cerr << "[ERR] Reading input file" << std::endl;
        return 1;
    }

    int numSamples = sfInfo.frames * sfInfo.channels;
    //capire come allineare i dati rispeto ai singoli warp 
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

    fftwf_execute(forwardPlan); //FFT  del segnale 

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
