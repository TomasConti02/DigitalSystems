#include <iostream>
#include <vector>
#include <cmath>
#include <sndfile.h>
#include <fftw3.h>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#define SAMPLE_RATE 44100  // Sample rate (Hz)

/*
    Band-Pass Filter

    HOW TO COMPILE: g++ -o bpf bpf.cpp -lfftw3 -lsndfile -msse2 -O3
           RUN:     ./bpf <lower-bound [Hz]> <upper-bound [Hz]>
*/

u_int64_t applyBandPassFilterFFTParallel(double* real, double* imag, int numSamples, double lowCutoff, double highCutoff, int sampleRate, std::ofstream& logFile, int index) {
    int lowCutoffIndex = static_cast<int>(lowCutoff / (static_cast<double>(sampleRate) / numSamples));
    int highCutoffIndex = static_cast<int>(highCutoff / (static_cast<double>(sampleRate) / numSamples));
    __m128d zero_register = _mm_setzero_pd();
    u_int64_t clock_counter_start = __rdtsc();
    
    // HPF
    for (int i = 0; i < lowCutoffIndex; i += 2) {
        _mm_storeu_pd(&real[i], zero_register);
        _mm_storeu_pd(&imag[i], zero_register);
    }

    // LPF
    for (int i = highCutoffIndex; i < numSamples / 2; i += 2) {
        _mm_storeu_pd(&real[i], zero_register);
        _mm_storeu_pd(&imag[i], zero_register);
    }

    u_int64_t clock_counter_end = __rdtsc();
    logFile << index << ", " << clock_counter_end - clock_counter_start << "\n";
    return clock_counter_end - clock_counter_start;
}

u_int64_t applyBandPassFilterFFTSequential(double* real, double* imag, int numSamples, double lowCutoff, double highCutoff, int sampleRate, std::ofstream& logFile, int index) {
    int lowCutoffIndex = static_cast<int>(lowCutoff / (static_cast<double>(sampleRate) / numSamples));
    int highCutoffIndex = static_cast<int>(highCutoff / (static_cast<double>(sampleRate) / numSamples));
    u_int64_t clock_counter_start = __rdtsc();
    
    for (int i = 0; i < lowCutoffIndex; ++i) {
        real[i] = 0.0;
        imag[i] = 0.0;
    }

    for (int i = highCutoffIndex; i < numSamples - highCutoffIndex; ++i) {
        real[i] = 0.0;
        imag[i] = 0.0;
    }

    u_int64_t clock_counter_end = __rdtsc();
    logFile << index << ", " << clock_counter_end - clock_counter_start << "\n";
    return clock_counter_end - clock_counter_start;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <low cutoff frequency in Hz> <high cutoff frequency in Hz>" << std::endl;
        return 1;
    }

    double lowCutoffFreq = std::stod(argv[1]);
    double highCutoffFreq = std::stod(argv[2]);

    if (lowCutoffFreq >= highCutoffFreq) {
        std::cerr << "Error: Low cutoff frequency must be less than high cutoff frequency" << std::endl;
        return 1;
    }

    if (lowCutoffFreq < 0 || highCutoffFreq > SAMPLE_RATE/2) {
        std::cerr << "Error: Cutoff frequencies must be between 0 and " << SAMPLE_RATE/2 << " Hz" << std::endl;
        return 1;
    }

    const char* inputFile = "./samples/drums.wav";
    const char* outputFile = "./samples/drumsBPF.wav";  // BPF = Band Pass Filter
    SF_INFO sfInfo;
    SNDFILE* inFile = sf_open(inputFile, SFM_READ, &sfInfo);

    if (!inFile) {
        std::cerr << "[ERR] in file " << inputFile << std::endl;
        return 1;
    }

    int numSamples = sfInfo.frames * sfInfo.channels;

    double* real = (double*)std::aligned_alloc(16, sizeof(double) * numSamples);
    if (real == nullptr) {
        std::cerr << "[ERR] Memory allocation failed for real array!" << std::endl;
        return 1;
    }

    double* imag = (double*)std::aligned_alloc(16, sizeof(double) * numSamples);
    if (imag == nullptr) {
        std::cerr << "[ERR] Memory allocation failed for imag array!" << std::endl;
        std::free(real);
        return 1;
    }

    std::vector<short> buffer(numSamples);
    sf_read_short(inFile, buffer.data(), numSamples);
    sf_close(inFile);

    for (int i = 0; i < numSamples; ++i) {
        real[i] = static_cast<double>(buffer[i]);
    }

    std::memset(imag, 0, sizeof(double) * numSamples);

    fftw_complex* fftData = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * numSamples);
    fftw_complex* ifftData = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * numSamples);
    fftw_plan forwardPlan = fftw_plan_dft_r2c_1d(numSamples, real, fftData, FFTW_ESTIMATE);
    fftw_plan inversePlan = fftw_plan_dft_c2r_1d(numSamples, fftData, real, FFTW_ESTIMATE);

    fftw_execute(forwardPlan);

    for (int i = 0; i < numSamples / 2 + 1; ++i) {
        real[i] = fftData[i][0];
        imag[i] = fftData[i][1];
    }

    // create log directory
    if (system("mkdir -p log") != 0) {
        std::cerr << "[ERR] Could not create log directory!" << std::endl;
        return 1;
    }

    std::ofstream logParallel("log/bpf_parallel.txt");
    std::ofstream logSequential("log/bpf_sequential.txt");
    std::ofstream SpeedUp("log/bpf_speedup.txt");

    if (!logParallel.is_open() || !logSequential.is_open() || !SpeedUp.is_open()) {
        std::cerr << "[ERR] Could not open log files!" << std::endl;
        return 1;
    }

    u_int64_t x;
    u_int64_t y;
    for (int i = 0; i < 30; ++i) {
        x = applyBandPassFilterFFTSequential(real, imag, numSamples / 2, lowCutoffFreq, highCutoffFreq, SAMPLE_RATE, logSequential, i + 1);
        y = applyBandPassFilterFFTParallel(real, imag, numSamples / 2, lowCutoffFreq, highCutoffFreq, SAMPLE_RATE, logParallel, i + 1);
        SpeedUp << (i++) << ", " << static_cast<double>(x) / static_cast<double>(y) << "\n";
    }
    
    logParallel.close();
    logSequential.close();
    SpeedUp.close();

    for (int i = 0; i < numSamples / 2 + 1; ++i) {
        fftData[i][0] = real[i];
        fftData[i][1] = imag[i];
    }

    fftw_execute(inversePlan);

    double normalFactor = 1.0 / numSamples;
    for (int i = 0; i < numSamples; ++i) {
        real[i] *= normalFactor;
    }

    for (int i = 0; i < numSamples; ++i) {
        buffer[i] = static_cast<short>(std::round(real[i]));
    }

    SNDFILE* outFile = sf_open(outputFile, SFM_WRITE, &sfInfo);
    if (!outFile) {
        std::cerr << "[ERR] in file " << outputFile << std::endl;
        std::free(real);
        std::free(imag);
        return 1;
    }

    sf_write_short(outFile, buffer.data(), numSamples);
    sf_close(outFile);

    std::cout << "Band-Pass Filter applied with frequency range " << lowCutoffFreq << " Hz to " << highCutoffFreq << " Hz and saved in " << outputFile << std::endl;

    fftw_destroy_plan(forwardPlan);
    fftw_destroy_plan(inversePlan);
    fftw_free(fftData);
    fftw_free(ifftData);

    std::free(real);
    std::free(imag);

    return 0;
}
