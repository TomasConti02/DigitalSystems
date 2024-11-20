/*
    g++ -o ParallelEQ ParallelEQ.cpp -lsndfile -lfftw3 -lm -mavx512f
    g++ -o parallel_equalizer ParallelEQ.cpp -lsndfile -lfftw3f
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
void applyParallelEqualizerFloat(float* real, float* imag, int numSamples, int sampleRate) {
    uint64_t clock_counter_start = __rdtsc();

    // Calcolo dei limiti di banda in base alla frequenza di campionamento
    int lowBandEnd = static_cast<int>(300 / (static_cast<float>(sampleRate) / numSamples)); // fino a 300 Hz
    int midBandStart = lowBandEnd;
    int midBandEnd = static_cast<int>(3000 / (static_cast<float>(sampleRate) / numSamples)); // da 300 a 3000 Hz
    int highBandStart = midBandEnd;

    // Guadagni per le bande
    __m128 lowGain = _mm_set1_ps(std::pow(10.0f, -6.0f / 20.0f));  // -6 dB
    __m128 midGain = _mm_set1_ps(std::pow(10.0f, 2.0f / 20.0f));   // +2 dB
    __m128 highGain = _mm_set1_ps(std::pow(10.0f, -3.0f / 20.0f)); // -3 dB

    // Filtraggio parallelo
    for (int i = 0; i < lowBandEnd; i += 4) {
        // Carica 4 valori reali e immaginari, moltiplica per il guadagno basso, e salva
        _mm_store_ps(&real[i], _mm_mul_ps(lowGain, _mm_load_ps(&real[i])));
        _mm_store_ps(&imag[i], _mm_mul_ps(lowGain, _mm_load_ps(&imag[i])));
    }

    for (int i = midBandStart/4; i < midBandEnd/4; i ++) {
        // Carica 4 valori reali e immaginari, moltiplica per il guadagno medio, e salva
        _mm_store_ps(&real[4*i], _mm_mul_ps(midGain, _mm_load_ps(&real[4*i])));
        _mm_store_ps(&imag[4*i], _mm_mul_ps(midGain, _mm_load_ps(&imag[4*i])));
    }

    for (int i = highBandStart/4; i < numSamples/4; i ++) {
        // Carica 4 valori reali e immaginari, moltiplica per il guadagno alto, e salva
        _mm_store_ps(&real[4*i], _mm_mul_ps(highGain, _mm_load_ps(&real[4*i])));
        _mm_store_ps(&imag[4*i], _mm_mul_ps(highGain, _mm_load_ps(&imag[4*i])));
    }

    uint64_t clock_counter_end = __rdtsc();
    printf("Time: %li\n", clock_counter_end - clock_counter_start);
}
int main(int argc, char* argv[]) {
    const char* inputFile = "/home/tomas/Desktop/noise.wav";
    const char* outputFile = "/home/tomas/Desktop/NOISETOMASfullSongParallel1EQFloat.wav";
    SF_INFO sfInfo;
    SNDFILE* inFile = sf_open(inputFile, SFM_READ, &sfInfo);

    if (!inFile) {
        std::cerr << "[ERR] in file " << inputFile << std::endl;
        return 1;
    }

    int numSamples = sfInfo.frames * sfInfo.channels;

    // Usa float invece di double
    float* real = (float*)std::aligned_alloc(16, sizeof(float) * numSamples);
    float* imag = (float*)std::aligned_alloc(16, sizeof(float) * numSamples);

    std::vector<short> buffer(numSamples);

    sf_read_short(inFile, buffer.data(), numSamples);
    sf_close(inFile);

    for (int i = 0; i < numSamples; ++i) {
        real[i] = static_cast<float>(buffer[i]);
    }
    std::memset(imag, 0, sizeof(float) * numSamples);

    // FFTW per precisione singola
    fftwf_complex* fftData = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * numSamples);
    fftwf_complex* ifftData = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * numSamples);
    fftwf_plan forwardPlan = fftwf_plan_dft_r2c_1d(numSamples, real, fftData, FFTW_ESTIMATE);
    fftwf_plan inversePlan = fftwf_plan_dft_c2r_1d(numSamples, fftData, real, FFTW_ESTIMATE);

    // FFT
    fftwf_execute(forwardPlan);

    // Copia dati in due array
    for (int i = 0; i < numSamples / 2 + 1; ++i) {
        real[i] = fftData[i][0];
        imag[i] = fftData[i][1];
    }

    // Applica EQ
    applyParallelEqualizerFloat(real, imag, numSamples / 2, SAMPLE_RATE);

    // Copia dati per la IFFT
    for (int i = 0; i < numSamples / 2 + 1; ++i) {
        fftData[i][0] = real[i];
        fftData[i][1] = imag[i];
    }

    // IFFT
    fftwf_execute(inversePlan);

    // Normalizzazione
    float normalFactor = 1.0f / numSamples;
    for (int i = 0; i < numSamples; ++i) {
        real[i] *= normalFactor;
    }

    // Converte in short per salvare nel file di output
    for (int i = 0; i < numSamples; ++i) {
        buffer[i] = static_cast<short>(std::round(real[i]));
    }

    // Salva il file audio equalizzato
    SNDFILE* outFile = sf_open(outputFile, SFM_WRITE, &sfInfo);
    if (!outFile) {
        std::cerr << "[ERR] in file " << outputFile << std::endl;
        return 1;
    }

    sf_write_short(outFile, buffer.data(), numSamples);
    sf_close(outFile);

    std::cout << "File equalized saved in " << outputFile << std::endl;

    // Libera memoria
    fftwf_destroy_plan(forwardPlan);
    fftwf_destroy_plan(inversePlan);
    fftwf_free(fftData);
    fftwf_free(ifftData);
    std::free(real);
    std::free(imag);

    return 0;
}
// function to apply the equalization
/*
void applyParallelEqualizer(double* real, double* imag, int numSamples, int sampleRate) {
    u_int64_t clock_counter_start = __rdtsc();

    int lowBandEnd = static_cast<int>(300 / (static_cast<double>(sampleRate) / numSamples)); // up to 300 Hz
    int midBandStart = lowBandEnd;
    int midBandEnd = static_cast<int>(3000 / (static_cast<double>(sampleRate) / numSamples)); // from 300 to 3000 Hz
    int highBandStart = midBandEnd;

    // Gains
    __m512d lowGain = _mm512_set1_pd(std::pow(10, -6.0 / 20.0));  // -6 dB
    __m512d midGain = _mm512_set1_pd(std::pow(10, 2.0 / 20.0));   // +2 dB
    __m512d highGain = _mm512_set1_pd(std::pow(10, -3.0 / 20.0)); // -3 dB

    // Low band
    //print_m512d(_mm512_store_pd(&real[0], _mm512_mul_pd(lowGain, _mm512_load_pd(&real[0]))));
   // print_m512d(_mm512_store_pd(&imag[0], _mm512_mul_pd(lowGain, _mm512_load_pd(&imag[0]))));
    for (int i = 0; i < lowBandEnd; i += 8) {  // 8 double per ciclo
        _mm512_store_pd(&real[i], _mm512_mul_pd(lowGain, _mm512_load_pd(&real[i])));
        _mm512_store_pd(&imag[i], _mm512_mul_pd(lowGain, _mm512_load_pd(&imag[i])));
    }

    // Mid band
    for (int i = midBandStart / 8; i < midBandEnd / 8; ++i) {
        _mm512_store_pd(&real[8 * i], _mm512_mul_pd(midGain, _mm512_load_pd(&real[8 * i])));
        _mm512_store_pd(&imag[8 * i], _mm512_mul_pd(midGain, _mm512_load_pd(&imag[8 * i])));
    }

    // High band
    for (int i = highBandStart / 8; i < numSamples / 8; ++i) {
        _mm512_store_pd(&real[8 * i], _mm512_mul_pd(highGain, _mm512_load_pd(&real[8 * i])));
        _mm512_store_pd(&imag[8 * i], _mm512_mul_pd(highGain, _mm512_load_pd(&imag[8 * i])));
    }

    u_int64_t clock_counter_end = __rdtsc();
    printf("Time: %li\n", clock_counter_end - clock_counter_start);
}
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
*/
/*
void applyParallelEqualizer(double* real, double* imag, int numSamples, int sampleRate) {
    u_int64_t clock_counter_start = __rdtsc();

    int lowBandEnd = static_cast<int>(300 / (static_cast<double>(sampleRate) / numSamples)); // up to 300 Hz
    int midBandStart = lowBandEnd;
    int midBandEnd = static_cast<int>(3000 / (static_cast<double>(sampleRate) / numSamples)); // from 300 to 3000 Hz
    int highBandStart = midBandEnd;

    // Gains
    __m256d lowGain = _mm256_set1_pd(std::pow(10, -6.0 / 20.0));  // -6 dB
    __m256d midGain = _mm256_set1_pd(std::pow(10, 2.0 / 20.0));   // +2 dB
    __m256d highGain = _mm256_set1_pd(std::pow(10, -3.0 / 20.0)); // -3 dB

    // Low band
    for (int i = 0; i < lowBandEnd; i += 4) {
        __m256d realVals = _mm256_load_pd(&real[i]);
        __m256d imagVals = _mm256_load_pd(&imag[i]);
        _mm256_store_pd(&real[i], _mm256_mul_pd(lowGain, realVals));
        _mm256_store_pd(&imag[i], _mm256_mul_pd(lowGain, imagVals));
    }

    // Mid band
    for (int i = midBandStart / 4; i < midBandEnd / 4; ++i) {
        __m256d realVals = _mm256_load_pd(&real[4 * i]);
        __m256d imagVals = _mm256_load_pd(&imag[4 * i]);
        _mm256_store_pd(&real[4 * i], _mm256_mul_pd(midGain, realVals));
        _mm256_store_pd(&imag[4 * i], _mm256_mul_pd(midGain, imagVals));
    }

    // High band
    for (int i = highBandStart / 4; i < numSamples / 4; ++i) {
        __m256d realVals = _mm256_load_pd(&real[4 * i]);
        __m256d imagVals = _mm256_load_pd(&imag[4 * i]);
        _mm256_store_pd(&real[4 * i], _mm256_mul_pd(highGain, realVals));
        _mm256_store_pd(&imag[4 * i], _mm256_mul_pd(highGain, imagVals));
    }

    u_int64_t clock_counter_end = __rdtsc();
    printf("Time: %li\n", clock_counter_end - clock_counter_start);
}
*/
