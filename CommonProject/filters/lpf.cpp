/*
    Low-Pass Filter for .wav files
    Libraries required: lsndfile, lfftw3

    HOW TO COMPILE: g++ lpf.cpp -o lpf -msse2 -lsndfile -lfftw3

    HOW TO RUN: ./lpf <cutoff_freq> <input_file>
       example: ./lpf 1000 ./samples/fullSong1.wav

    The output is the speedup (scalar time / parallel time)
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <sndfile.h>
#include <fftw3.h>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#define SAMPLE_RATE 44100  // Sample rate (Hz)

u_int64_t applyLowPassFilterFFTParallel(double* real, double* imag, int numSamples, double cutoffFreq, int sampleRate) {
    int cutoffIndex = static_cast<int>(cutoffFreq / (static_cast<double>(sampleRate) / numSamples));
    u_int64_t clock_counter_start = __rdtsc();
    __m128d zero_register = _mm_setzero_pd();
    for (int i = cutoffIndex / 2; i < numSamples / 2; i++) {
        _mm_store_pd(&real[i * 2], zero_register);
        _mm_store_pd(&imag[i * 2], zero_register);
    }
    u_int64_t clock_counter_end = __rdtsc();
    return clock_counter_end - clock_counter_start;
}

u_int64_t applyLowPassFilterFFTSequential(double* real, double* imag, int numSamples, double cutoffFreq, int sampleRate) {
    int cutoffIndex = static_cast<int>(cutoffFreq / (static_cast<double>(sampleRate) / numSamples));
    u_int64_t clock_counter_start = __rdtsc();
    for (int i = cutoffIndex; i < numSamples; ++i) {
        real[i] = 0.0;
        imag[i] = 0.0;
    }
    u_int64_t clock_counter_end = __rdtsc();
    return clock_counter_end - clock_counter_start;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <cutoff frequency in Hz> <input file path>" << std::endl;
        return 1;
    }

    double cutoffFreq = std::stod(argv[1]);
    const char* inputFile = argv[2];

    SF_INFO sfInfo;
    SNDFILE* inFile = sf_open(inputFile, SFM_READ, &sfInfo);
    if (!inFile) {
        std::cerr << "[ERR] Could not open file " << inputFile << std::endl;
        return 1;
    }

    int numSamples = sfInfo.frames * sfInfo.channels;
    double* real = (double*)std::aligned_alloc(16, sizeof(double) * numSamples);
    double* imag = (double*)std::aligned_alloc(16, sizeof(double) * numSamples);
    double* realParallel = (double*)std::aligned_alloc(16, sizeof(double) * numSamples);
    double* imagParallel = (double*)std::aligned_alloc(16, sizeof(double) * numSamples);

    std::vector<short> buffer(numSamples);
    sf_read_short(inFile, buffer.data(), numSamples);
    sf_close(inFile);

    for (int i = 0; i < numSamples; ++i) {
        real[i] = realParallel[i] = static_cast<double>(buffer[i]);
    }

    std::memset(imag, 0, sizeof(double) * numSamples);
    std::memset(imagParallel, 0, sizeof(double) * numSamples);

    fftw_plan forwardPlan = fftw_plan_dft_r2c_1d(numSamples, real, reinterpret_cast<fftw_complex*>(real), FFTW_ESTIMATE);
    fftw_plan forwardPlanParallel = fftw_plan_dft_r2c_1d(numSamples, realParallel, reinterpret_cast<fftw_complex*>(realParallel), FFTW_ESTIMATE);

    fftw_execute(forwardPlan);
    fftw_execute(forwardPlanParallel);

    u_int64_t x, y;
    x = applyLowPassFilterFFTSequential(real, imag, numSamples / 2, cutoffFreq, SAMPLE_RATE);
    y = applyLowPassFilterFFTParallel(realParallel, imagParallel, numSamples / 2, cutoffFreq, SAMPLE_RATE);

    std::cout << (float) 1.0 * ((float) x / (float) y) << std::endl;

    fftw_destroy_plan(forwardPlan);
    fftw_destroy_plan(forwardPlanParallel);
    std::free(real);
    std::free(imag);
    std::free(realParallel);
    std::free(imagParallel);

    return 0;
}
