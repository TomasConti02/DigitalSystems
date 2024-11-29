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

const float LOW_GAIN = -60.0f;
const float MID_GAIN = 2.0f;
const float HIGH_GAIN = -3.0f;

__global__ void applyEqualizerKernel(float* real, float* imag, int numSamples, int lowBandEnd, 
                                    int midBandEnd, float lowGain, float midGain, float highGain) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numSamples) {
        float gainFactor;
        
        if (idx < lowBandEnd) {
            gainFactor = lowGain;
        } else if (idx < midBandEnd) {
            gainFactor = midGain;
        } else {
            gainFactor = highGain;
        }
        
        real[idx] *= gainFactor;
        imag[idx] *= gainFactor;
    }
}

void applyCudaEqualizer(float* real, float* imag, int numSamples, int sampleRate) {
    float *d_real, *d_imag;
    int lowBandEnd = static_cast<int>(300.0f / (static_cast<float>(sampleRate) / numSamples));
    int midBandEnd = static_cast<int>(3000.0f / (static_cast<float>(sampleRate) / numSamples));
    
    float lowGain = std::pow(10.0f, LOW_GAIN / 20.0f);
    float midGain = std::pow(10.0f, MID_GAIN / 20.0f);
    float highGain = std::pow(10.0f, HIGH_GAIN / 20.0f);

    cudaMalloc(&d_real, numSamples * sizeof(float));
    cudaMalloc(&d_imag, numSamples * sizeof(float));
    
    cudaMemcpy(d_real, real, numSamples * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_imag, imag, numSamples * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((numSamples + BLOCK_SIZE - 1) / BLOCK_SIZE);
    applyEqualizerKernel<<<gridDim, blockDim>>>(d_real, d_imag, numSamples, 
                                               lowBandEnd, midBandEnd,
                                               lowGain, midGain, highGain);
    
    cudaMemcpy(real, d_real, numSamples * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(imag, d_imag, numSamples * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_real);
    cudaFree(d_imag);
}

int main(int argc, char* argv[]) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
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
    float* real = (float*)std::aligned_alloc(64, sizeof(float) * numSamples);
    float* imag = (float*)std::aligned_alloc(64, sizeof(float) * numSamples);
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

    cudaEventRecord(start);
    applyCudaEqualizer(real, imag, numSamples / 2, SAMPLE_RATE);
    cudaEventRecord(stop);
    
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

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "CUDA Execution time: " << milliseconds << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
