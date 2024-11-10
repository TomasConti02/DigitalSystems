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
    int cutoffIndex = static_cast<int>(cutoffFreq / (static_cast<double>(sampleRate) / numSamples));
    __m128d zero_register = _mm_setzero_pd();
    u_int64_t clock_counter_start = __rdtsc();
    for (int i = cutoffIndex; i < numSamples / 2; ++i) {
        _mm_storeu_pd(&real[2 * i], zero_register);  // Azzera 2 valori reali
        _mm_storeu_pd(&imag[2 * i], zero_register);  // Azzera 2 valori immaginari
    }

    u_int64_t clock_counter_end = __rdtsc();
    logFile << index << ", " << clock_counter_end - clock_counter_start<< "\n";  // Log dell'indice e dei cicli di clock
    return clock_counter_end - clock_counter_start;
}

u_int64_t applyLowPassFilterFFTSequential(double* real, double* imag, int numSamples, double cutoffFreq, int sampleRate, std::ofstream& logFile, int index) {
    int cutoffIndex = static_cast<int>(cutoffFreq / (static_cast<double>(sampleRate) / numSamples));
     u_int64_t clock_counter_start = __rdtsc();
    for (int i = cutoffIndex; i < numSamples - cutoffIndex; ++i) {
        real[i] = 0.0;
        imag[i] = 0.0;
    }

    u_int64_t clock_counter_end = __rdtsc();
    logFile << index << ", " << clock_counter_end - clock_counter_start<< "\n";  // Log dell'indice e dei cicli di clock
    return clock_counter_end - clock_counter_start;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <cutoff frequency in Hz>" << std::endl;
        return 1;
    }

    double cutoffFreq = std::stod(argv[1]);

    const char* inputFile = "/home/tomas/Desktop/drums.wav";
    const char* outputFile = "/home/tomas/Desktop/drumsLPF.wav";
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

    // File di log per il codice parallelo e sequenziale
    std::ofstream logParallel("log_parallel.txt");
    std::ofstream logSequential("log_sequential.txt");
    std::ofstream SpeedUp("SpeedUp.txt");

    if (!logParallel.is_open() || !logSequential.is_open() || !SpeedUp.is_open()) {
        std::cerr << "[ERR] Could not open log files!" << std::endl;
        return 1;
    }

    // Ciclo 10 volte per il codice parallelo e sequenziale
    u_int64_t x;
    u_int64_t y;
    for (int i = 0; i < 30; ++i) {
        x=applyLowPassFilterFFTSequential(real, imag, numSamples / 2, cutoffFreq, SAMPLE_RATE, logSequential, i + 1);
        y=applyLowPassFilterFFTParallel(real, imag, numSamples / 2, cutoffFreq, SAMPLE_RATE, logParallel, i + 1);
        SpeedUp << (i++) << ", " << static_cast<double>(x) / static_cast<double>(y)<< "\n";  // Log dell'indice e dei cicli di clock
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

    std::cout << "Low-Pass Filter applied with cutoff frequency " << cutoffFreq << " Hz and saved in " << outputFile << std::endl;

    fftw_destroy_plan(forwardPlan);
    fftw_destroy_plan(inversePlan);
    fftw_free(fftData);
    fftw_free(ifftData);

    std::free(real);
    std::free(imag);

    return 0;
}


/*
// Real and imaginary parts are stored in two separate arrays
void applyLowPassFilterFFT(double* real, double* imag, int numSamples, double cutoffFreq, int sampleRate) {
    int cutoffIndex = static_cast<int>(cutoffFreq / (static_cast<double>(sampleRate) / numSamples));
    __m128d zero_register = _mm_setzero_pd();
    // Utilizzare i registri SIMD (__m128d) per lavorare in blocchi da 2 valori per volta
    __m128d* RealRegistry = (__m128d*) real;
    __m128d* ImagRegistry = (__m128d*) imag;
    __m128d realReg;
    __m128d imagReg ;
    // Elaborazione dei dati in blocchi di 2 valori
    for (int i = 0; i < numSamples / 2; ++i) {
        // Carica i dati nei registri SIMD (__m128d)
        realReg = _mm_loadu_pd(&real[2*i]);  // Carica 2 valori reali (double)
        imagReg = _mm_loadu_pd(&imag[2*i]);  // Carica 2 valori immaginari (double)

        // Se l'indice supera la frequenza di cutoff, azzera i dati
        if (i >= cutoffIndex) {
            realReg = _mm_setzero_pd();  // Azzeramento dei valori reali
            imagReg = _mm_setzero_pd();  // Azzeramento dei valori immaginari
        }

        // Salva i dati modificati nei registri
        _mm_storeu_pd(&real[2*i], realReg);  // Salva i dati modificati nei real
        _mm_storeu_pd(&imag[2*i], imagReg);  // Salva i dati modificati negli immaginari
    }
    for (int i = cutoffIndex; i < ((numSamples - cutoffIndex)/2); ++i) {
        realReg = _mm_loadu_pd(&real[2*i]);  // Carica 2 valori reali (double)
        imagReg = _mm_loadu_pd(&imag[2*i]);  // Carica 2 valori immaginari (double)
        _mm_storeu_pd(&real[2*i], _mm_and_pd(realReg, zero_register));  // Salva i dati modificati nei real
        _mm_storeu_pd(&imag[2*i], _mm_and_pd(imagReg, zero_register));  // Salva i dati modificati negli immaginari

    }

}

void applyLowPassFilterFFT(double* real, double* imag, int numSamples, double cutoffFreq, int sampleRate) {
    int cutoffIndex = static_cast<int>(cutoffFreq / (static_cast<double>(sampleRate) / numSamples));
    for (int i = cutoffIndex; i < numSamples - cutoffIndex; ++i) {
        real[i] = 0.0;
        imag[i] = 0.0;
    }
}*/
