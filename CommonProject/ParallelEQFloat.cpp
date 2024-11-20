/*
    g++ -o ParallelEQ ParallelEQ.cpp -lsndfile -lfftw3f -msse2

    This equalizer divides the frequency spectrum into three bands, and for each one applies a gain:

    [0 Hz - 300 Hz]     [300 Hz - 3000 Hz]     [3000 Hz - 22 kHz]
          -6 dB                 +2 dB                   -3 dB

*/

#include <iostream>
#include <vector>
#include <cmath>
#include <sndfile.h>
#include <fftw3.h>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#define SAMPLE_RATE 44100  // Sample rate (Hz)

// function to print the register
void print_m128(__m128 reg, const std::string& label = "") {
    alignas(16) float values[2]; // aligned array
    _mm_storeu_ps(values, reg);   // stores register on array
    
    if (!label.empty()) {
        std::cout << label << ": ";
    }
    std::cout << "[" << values[1] << ", " << values[0] << "]" << std::endl; // print
}

// parallel EQ
u_int64_t applyParallelEqualizerFloat(float* real, float* imag, int numSamples, int sampleRate) {
    uint64_t clock_counter_start = __rdtsc();

    // computes the bands
    int lowBandEnd = static_cast<int>(300 / (static_cast<float>(sampleRate) / numSamples)); // up to 300 Hz
    int midBandStart = lowBandEnd;
    int midBandEnd = static_cast<int>(3000 / (static_cast<float>(sampleRate) / numSamples)); // from 300 to 3000 Hz
    int highBandStart = midBandEnd;

    // gain
    __m128 lowGain = _mm_set1_ps(std::pow(10.0f, -6.0f / 20.0f));  // -6 dB
    __m128 midGain = _mm_set1_ps(std::pow(10.0f, 2.0f / 20.0f));   // +2 dB
    __m128 highGain = _mm_set1_ps(std::pow(10.0f, -3.0f / 20.0f)); // -3 dB

    // parallel filter
    for (int i = 0; i < lowBandEnd; i += 4) {
        // load of 4 values
        _mm_store_ps(&real[i], _mm_mul_ps(lowGain, _mm_load_ps(&real[i])));
        _mm_store_ps(&imag[i], _mm_mul_ps(lowGain, _mm_load_ps(&imag[i])));
    }

    for (int i = midBandStart/4; i < midBandEnd/4; i ++) {
        _mm_store_ps(&real[4*i], _mm_mul_ps(midGain, _mm_load_ps(&real[4*i])));
        _mm_store_ps(&imag[4*i], _mm_mul_ps(midGain, _mm_load_ps(&imag[4*i])));
    }

    for (int i = highBandStart/4; i < numSamples/4; i ++) {
        _mm_store_ps(&real[4*i], _mm_mul_ps(highGain, _mm_load_ps(&real[4*i])));
        _mm_store_ps(&imag[4*i], _mm_mul_ps(highGain, _mm_load_ps(&imag[4*i])));
    }

    uint64_t clock_counter_end = __rdtsc();
    return (clock_counter_end - clock_counter_start);
}

// scalar EQ
uint64_t applyScalarEqualizerFloat(float* real, float* imag, int numSamples, int sampleRate) {
    uint64_t clock_counter_start = __rdtsc();

    // compute bands
    int lowBandEnd = static_cast<int>(300 / (static_cast<float>(sampleRate) / numSamples)); // up to 300 Hz
    int midBandStart = lowBandEnd;
    int midBandEnd = static_cast<int>(3000 / (static_cast<float>(sampleRate) / numSamples)); // from 300 to 3000 Hz
    int highBandStart = midBandEnd;

    // gain
    float lowGain = std::pow(10.0f, -6.0f / 20.0f);  // -6 dB
    float midGain = std::pow(10.0f, 2.0f / 20.0f);   // +2 dB
    float highGain = std::pow(10.0f, -3.0f / 20.0f); // -3 dB

    // filter
    for (int i = 0; i < lowBandEnd; ++i) {
        real[i] *= lowGain;
        imag[i] *= lowGain;
    }

    for (int i = midBandStart; i < midBandEnd; ++i) {
        real[i] *= midGain;
        imag[i] *= midGain;
    }

    for (int i = highBandStart; i < numSamples; ++i) {
        real[i] *= highGain;
        imag[i] *= highGain;
    }

    uint64_t clock_counter_end = __rdtsc();
    return (clock_counter_end - clock_counter_start);
}



int main(int argc, char* argv[]) {
    const char* inputFile = "./samples/fullSong1.wav";
    const char* outputFile = "./samples/fullSong1EQ.wav";
    SF_INFO sfInfo;
    SNDFILE* inFile = sf_open(inputFile, SFM_READ, &sfInfo);

    if (!inFile) {
        std::cerr << "[ERR] in file " << inputFile << std::endl;
        return 1;
    }

    int numSamples = sfInfo.frames * sfInfo.channels;

    float* real = (float*)std::aligned_alloc(64, sizeof(float) * numSamples);
    float* imag = (float*)std::aligned_alloc(64, sizeof(float) * numSamples);
    std::vector<short> buffer(numSamples);

    sf_read_short(inFile, buffer.data(), numSamples);
    sf_close(inFile);

    for (int i = 0; i < numSamples; ++i) {
        real[i] = static_cast<float>(buffer[i]);
    }
    std::memset(imag, 0, sizeof(float) * numSamples);
    // FFT & IFFT
    fftwf_complex* fftData = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * numSamples);
    fftwf_complex* ifftData = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * numSamples);
    fftwf_plan forwardPlan = fftwf_plan_dft_r2c_1d(numSamples, real, fftData, FFTW_ESTIMATE);
    fftwf_plan inversePlan = fftwf_plan_dft_c2r_1d(numSamples, fftData, real, FFTW_ESTIMATE);

    // FFT
    fftwf_execute(forwardPlan);

    // copy data in two arrays
    for (int i = 0; i < numSamples / 2 + 1; ++i) {
        real[i] = fftData[i][0];
        imag[i] = fftData[i][1];
    }

    // applies EQ
    unsigned long long parallelTime = applyParallelEqualizerFloat(real, imag, numSamples / 2, SAMPLE_RATE);
    unsigned long long scalarTime = applyScalarEqualizerFloat(real, imag, numSamples / 2, SAMPLE_RATE);

    // prints the speedup
    printf("\tSPEEDUP: %f\n", (float) scalarTime / (float) parallelTime);

    // copy data to prepare the Inverse FFT
    for (int i = 0; i < numSamples / 2 + 1; ++i) {
        fftData[i][0] = real[i];
        fftData[i][1] = imag[i];
    }

    // IFFT
    fftwf_execute(inversePlan);

    // normalization
    double normalFactor = 1.0 / numSamples;
    for (int i = 0; i < numSamples; ++i) {
        real[i] *= normalFactor;
    }

    // converts to short to save on out file
    for (int i = 0; i < numSamples; ++i) {
        buffer[i] = static_cast<short>(std::round(real[i]));
    }

    // saves equalized audio file
    SNDFILE* outFile = sf_open(outputFile, SFM_WRITE, &sfInfo);
    if (!outFile) {
        std::cerr << "[ERR] in file " << outputFile << std::endl;
        return 1;
    }

    sf_write_short(outFile, buffer.data(), numSamples);
    sf_close(outFile);

    // free
    fftwf_destroy_plan(forwardPlan);
    fftwf_destroy_plan(inversePlan);
    fftwf_free(fftData);
    fftwf_free(ifftData);

    return 0;
}
