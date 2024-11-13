//g++ -o scalarLPF Music.cpp -lsndfile -lfftw3 -lm
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

u_int64_t applyLowPassFilterFFTParallel(double* real, double* imag, int numSamples, double cutoffFreq, int sampleRate, std::ofstream& logFile, int index) {
    std::cout << "Low-Pass Filter applied with cutoff frequency CASO PARALLELO" << std::endl;
    int cutoffIndex = static_cast<int>(cutoffFreq / (static_cast<double>(sampleRate) / numSamples));
    std::cout << "cutoffIndex: "<<cutoffIndex<<" limite ciclo : "<<(numSamples / 2)<< std::endl;
    u_int64_t clock_counter_start = __rdtsc();
    __m128d zero_register = _mm_setzero_pd();
    for (int i = cutoffIndex; i < numSamples / 2; ++i) {
        _mm_store_pd(&real[2 * i], zero_register);  // Azzera 2 valori reali
        _mm_store_pd(&imag[2 * i], zero_register);  // Azzera 2 valori immaginari
    }

    u_int64_t clock_counter_end = __rdtsc();
    logFile << index << ", " << clock_counter_end - clock_counter_start<< "\n";  // Log dell'indice e dei cicli di clock
    return clock_counter_end - clock_counter_start;
}

u_int64_t applyLowPassFilterFFTSequential(double* real, double* imag, int numSamples, double cutoffFreq, int sampleRate, std::ofstream& logFile, int index) {
    std::cout << "Low-Pass Filter applied with cutoff frequency CASO SEQUEZIALE" << std::endl;
    int cutoffIndex = static_cast<int>(cutoffFreq / (static_cast<double>(sampleRate) / numSamples));
    std::cout << "cutoffIndex: "<<cutoffIndex<<" limite ciclo : "<<(numSamples - cutoffIndex)<< std::endl;
     u_int64_t clock_counter_start = __rdtsc();
    for (int i = cutoffIndex; i < numSamples - cutoffIndex; ++i) {
        real[i] = 0.0;
        imag[i] = 0.0;
    }

    u_int64_t clock_counter_end = __rdtsc();
    logFile << index << ", " << clock_counter_end - clock_counter_start<< "\n";  // Log dell'indice e dei cicli di clock
    return clock_counter_end - clock_counter_start;
}
/*
u_int64_t ProvaLowPassFilterFFTP(double* real, double* imag, int numSamples, double cutoffFreq, int sampleRate){
    int cutoffIndex = static_cast<int>(cutoffFreq / (static_cast<double>(sampleRate) / numSamples));
    u_int64_t clock_counter_start = __rdtsc();
    __m128d zero_register = _mm_setzero_pd();
    __m128d real_values, __m128d;
    __m128d* p_real = (__m128d*) real;
    __m128d* p_img = (__m128d*) imag;
    for (int i = cutoffIndex; i < numSamples / 2; ++i) {
         real_values = _mm_load_pd((double*) &p_real[2*i]);
         imag_values = _mm_load_pd((double*) &p_img[2*i]);
        // Azzeramento dei valori con un AND
        real_values = _mm_and_pd(real_values, zero_register);
        imag_values = _mm_and_pd(imag_values, zero_register);
    }
    u_int64_t clock_counter_end = __rdtsc();
    logFile << index << ", " << clock_counter_end - clock_counter_start<< "\n";  // Log dell'indice e dei cicli di clock
    return clock_counter_end - clock_counter_start;
}*/
int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <cutoff frequency in Hz>" << std::endl;
        return 1;
    }

    double cutoffFreq = std::stod(argv[1]);

    const char* inputFile = "/home/tomas/Desktop/drums.wav";
    const char* outputFile = "/home/tomas/Desktop/drumsLPFSequenziale.wav";
    const char* outputFileParallelo = "/home/tomas/Desktop/drumsLPFParallelo.wav";
    SF_INFO sfInfo;
    SNDFILE* inFile = sf_open(inputFile, SFM_READ, &sfInfo);

    if (!inFile) {
        std::cerr << "[ERR] in file " << inputFile << std::endl;
        return 1;
    }

    int numSamples = sfInfo.frames * sfInfo.channels;
    double* real = (double*)std::aligned_alloc(16, sizeof(double) * numSamples);
    double* imag = (double*)std::aligned_alloc(16, sizeof(double) * numSamples);
    double* realParallelo = (double*)std::aligned_alloc(16, sizeof(double) * numSamples);
    double* imagParallelo = (double*)std::aligned_alloc(16, sizeof(double) * numSamples);

    std::vector<short> buffer(numSamples);
    std::vector<short> bufferParallelo(numSamples);
    sf_read_short(inFile, buffer.data(), numSamples);
    sf_close(inFile);

    for (int i = 0; i < numSamples; ++i) {
        real[i] = static_cast<double>(buffer[i]);
        realParallelo[i] = static_cast<double>(buffer[i]);
    }

    std::memset(imag, 0, sizeof(double) * numSamples);
    std::memset(imagParallelo, 0, sizeof(double) * numSamples);

    fftw_complex* fftData = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * numSamples);
    fftw_complex* ifftData = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * numSamples);
    fftw_plan forwardPlan = fftw_plan_dft_r2c_1d(numSamples, real, fftData, FFTW_ESTIMATE);
    fftw_plan inversePlan = fftw_plan_dft_c2r_1d(numSamples, fftData, real, FFTW_ESTIMATE);

    fftw_complex* fftDataParallelo = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * numSamples);
    fftw_complex* ifftDataParallelo = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * numSamples);
    fftw_plan forwardPlanParallelo = fftw_plan_dft_r2c_1d(numSamples, realParallelo, fftDataParallelo, FFTW_ESTIMATE);
    fftw_plan inversePlanParallelo = fftw_plan_dft_c2r_1d(numSamples, fftDataParallelo, realParallelo, FFTW_ESTIMATE);

    fftw_execute(forwardPlan);
    fftw_execute(forwardPlanParallelo);
    for (int i = 0; i < numSamples / 2 + 1; ++i) {
        real[i] = fftData[i][0];
        imag[i] = fftData[i][1];
        realParallelo[i] = fftDataParallelo[i][0];
        imagParallelo[i] = fftDataParallelo[i][1];
        
    }
    std::ofstream logParallel("log_parallel.txt");
    std::ofstream logSequential("log_sequential.txt");
    std::ofstream SpeedUp("SpeedUp.txt");

    if (!logParallel.is_open() || !logSequential.is_open() || !SpeedUp.is_open()) {
        std::cerr << "[ERR] Could not open log files!" << std::endl;
        return 1;
    }

    u_int64_t x, y;
    for (int i = 0; i < 1; ++i) {
        x = applyLowPassFilterFFTSequential(real, imag, numSamples / 2, cutoffFreq, SAMPLE_RATE, logSequential, i + 1);
        y = applyLowPassFilterFFTParallel(realParallelo, imagParallelo, numSamples / 2, cutoffFreq, SAMPLE_RATE, logParallel, i + 1);
        SpeedUp << i + 1 << ", " << static_cast<double>(x) / static_cast<double>(y) << "\n";
    }

    logParallel.close();
    logSequential.close();
    SpeedUp.close();
    for (int i = 0; i < numSamples / 2 + 1; ++i) {
        fftData[i][0] = real[i];
        fftData[i][1] = imag[i];
        fftDataParallelo[i][0] = realParallelo[i];
        fftDataParallelo[i][1] = imagParallelo[i];
    }
    fftw_execute(inversePlan);
    fftw_execute(inversePlanParallelo);

    double normalFactor = 1.0 / numSamples;
    for (int i = 0; i < numSamples; ++i) {
        real[i] *= normalFactor;
        realParallelo[i] *= normalFactor;
    }

    for (int i = 0; i < numSamples; ++i) {
        buffer[i] = static_cast<short>(std::round(real[i]));
        bufferParallelo[i] = static_cast<short>(std::round(realParallelo[i]));
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

    SNDFILE* outFile2 = sf_open(outputFileParallelo, SFM_WRITE, &sfInfo);
    if (!outFile2) {
        std::cerr << "[ERR] in file " << outputFileParallelo << std::endl;
        std::free(realParallelo);
        std::free(imagParallelo);
        return 1;
    }

    sf_write_short(outFile2, bufferParallelo.data(), numSamples);
    sf_close(outFile2);

    fftw_destroy_plan(forwardPlan);
    fftw_destroy_plan(forwardPlanParallelo);
    fftw_destroy_plan(inversePlan);
    fftw_destroy_plan(inversePlanParallelo);
    fftw_free(fftData);
    fftw_free(fftDataParallelo);

    std::free(real);
    std::free(imag);
    std::free(realParallelo);
    std::free(imagParallelo);

    std::cout << "Low-Pass Filters applied and files saved.\n";
    return 0;
}
