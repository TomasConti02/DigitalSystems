/*
    This equalizer divides the frequency spectrum into three bands, and for each one applies a gain:

    [0 Hz - 300 Hz]     [300 Hz - 3000 Hz]     [3000 Hz - 22 kHz]
          -6 dB                 +2 dB                   -3 dB

    
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <sndfile.h>
#include <fftw3.h>
#include <immintrin.h>
#include <emmintrin.h>
void print_m128d(__m128d reg, const std::string& label = "") {
    alignas(16) double values[2]; // Array allineato per la compatibilità SIMD
    _mm_storeu_pd(values, reg);   // Carica i valori dal registro nell'array
    
    if (!label.empty()) {
        std::cout << label << ": ";
    }
    std::cout << "[" << values[1] << ", " << values[0] << "]" << std::endl; // Stampa in ordine umano
}
#define SAMPLE_RATE 44100  // Sample rate (Hz)
// function to apply the equalization
void applyParallelEqualizer(double* real, double* imag, int numSamples, int sampleRate) {
    u_int64_t clock_counter_start = __rdtsc();
    int lowBandEnd = static_cast<int>(300 / (static_cast<double>(sampleRate) / numSamples)); // up to 300 Hz
    int midBandStart = lowBandEnd;
    int midBandEnd = static_cast<int>(3000 / (static_cast<double>(sampleRate) / numSamples)); // from 300 to 3000 Hz
    int highBandStart = midBandEnd;
    
    __m128d lowGain = _mm_set1_pd(std::pow(10, -6.0 / 20.0));// -6 dB
    __m128d midGain = _mm_set1_pd(std::pow(10, 2.0 / 20.0));// +2 dB
    __m128d highGain = _mm_set1_pd(std::pow(10, -3.0 / 20.0)); // -3 dB
   // print_m128d(lowGain, "Low Gain");
    //print_m128d(midGain, "Mid Gain");
    //print_m128d(highGain, "High Gain");
    //_mm_store_pd(&real[ i], _mm_mul_pd(lowGain,_mm_load_pd(real)));  // Azzera 2 valori reali
    //_mm_store_pd(&imag[ i],  _mm_mul_pd(lowGain,_mm_load_pd(imag)));  // Azzera 2 valori immaginari

    for (int i = 0; i < lowBandEnd; i+=2) {
       // j++;
        _mm_store_pd(&real[ i], _mm_mul_pd(lowGain,_mm_load_pd(&real[i])));  // Azzera 2 valori reali
        _mm_store_pd(&imag[ i],  _mm_mul_pd(lowGain,_mm_load_pd(&imag[i])));  // Azzera 2 valori immaginari
    }
    for (int i = midBandStart/2; i < midBandEnd/2; ++i) {
       // j++;
        _mm_store_pd(&real[2 * i], _mm_mul_pd(midGain,_mm_load_pd(&real[2*i])));  // Azzera 2 valori reali
        _mm_store_pd(&imag[2 * i],  _mm_mul_pd(midGain,_mm_load_pd(&imag[2*i])));  // Azzera 2 valori immaginari
    }
    for (int i = highBandStart/2; i < numSamples / 2; ++i) {
        _mm_store_pd(&real[2 * i], _mm_mul_pd(highGain,_mm_load_pd(&real[2*i])));  // Azzera 2 valori reali
        _mm_store_pd(&imag[2 * i],  _mm_mul_pd(highGain,_mm_load_pd(&imag[2*i])));  // Azzera 2 valori immaginari
    }
    u_int64_t clock_counter_end = __rdtsc();

    printf("Time: %li\n", clock_counter_end - clock_counter_start);
}

int main(int argc, char* argv[]) {
    const char* inputFile = "/home/tomas/Desktop/fullSong1.wav";
    const char* outputFile = "/home/tomas/Desktop/TOMASfullSongParallel1EQ.wav";
    SF_INFO sfInfo;
    SNDFILE* inFile = sf_open(inputFile, SFM_READ, &sfInfo);

    if (!inFile) {
        std::cerr << "[ERR] in file " << inputFile << std::endl;
        return 1;
    }

    int numSamples = sfInfo.frames * sfInfo.channels;
    std::vector<double> real(numSamples);
    std::vector<double> imag(numSamples, 0.0);
    std::vector<short> buffer(numSamples);

    sf_read_short(inFile, buffer.data(), numSamples);
    sf_close(inFile);

    for (int i = 0; i < numSamples; ++i) {
        real[i] = static_cast<double>(buffer[i]);
    }

    // FFT & IFFT
    fftw_complex* fftData = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * numSamples);
    fftw_complex* ifftData = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * numSamples);
    fftw_plan forwardPlan = fftw_plan_dft_r2c_1d(numSamples, real.data(), fftData, FFTW_ESTIMATE);
    fftw_plan inversePlan = fftw_plan_dft_c2r_1d(numSamples, fftData, real.data(), FFTW_ESTIMATE);

    // FFT
    fftw_execute(forwardPlan);

    // copy data in two arrays
    for (int i = 0; i < numSamples / 2 + 1; ++i) {
        real[i] = fftData[i][0];
        imag[i] = fftData[i][1];
    }

    // applies EQ
    applyParallelEqualizer(real.data(), imag.data(), numSamples / 2, SAMPLE_RATE);

    // copy data to prepare the Inverse FFT
    for (int i = 0; i < numSamples / 2 + 1; ++i) {
        fftData[i][0] = real[i];
        fftData[i][1] = imag[i];
    }

    // IFFT
    fftw_execute(inversePlan);

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

    std::cout << "File equalized saved in " << outputFile << std::endl;

    // free
    fftw_destroy_plan(forwardPlan);
    fftw_destroy_plan(inversePlan);
    fftw_free(fftData);
    fftw_free(ifftData);

    return 0;
}
