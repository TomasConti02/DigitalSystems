/*in pratica ho allineato gli array reali e img a 16byte e poi ho usato dei registri appositi per leggerli */
#include <iostream>
#include <vector>
#include <cmath>
#include <sndfile.h>
#include <fftw3.h>
#include <fstream>
#include <cstdlib> // per aligned_alloc o posix_memalign
#include <cstring> // per memset
#include <immintrin.h> // Intrinseci di Intel x86

#define SAMPLE_RATE 44100  // Sample rate (Hz)

// Real and imaginary parts are stored in two separate arrays
void applyLowPassFilterFFT(double* real, double* imag, int numSamples, double cutoffFreq, int sampleRate) {
    int cutoffIndex = static_cast<int>(cutoffFreq / (static_cast<double>(sampleRate) / numSamples));

    // Utilizzare i registri SIMD (__m128d) per lavorare in blocchi da 2 valori per volta
    __m128d* RealRegistry = (__m128d*) real;
    __m128d* ImagRegistry = (__m128d*) imag;

    // Elaborazione dei dati in blocchi di 2 valori
    for (int i = 0; i < numSamples / 2; ++i) {
        // Carica i dati nei registri SIMD (__m128d)
        __m128d realReg = _mm_loadu_pd(&real[2*i]);  // Carica 2 valori reali (double)
        __m128d imagReg = _mm_loadu_pd(&imag[2*i]);  // Carica 2 valori immaginari (double)

        // Se l'indice supera la frequenza di cutoff, azzera i dati
        if (i >= cutoffIndex) {
            realReg = _mm_setzero_pd();  // Azzeramento dei valori reali
            imagReg = _mm_setzero_pd();  // Azzeramento dei valori immaginari
        }

        // Salva i dati modificati nei registri
        _mm_storeu_pd(&real[2*i], realReg);  // Salva i dati modificati nei real
        _mm_storeu_pd(&imag[2*i], imagReg);  // Salva i dati modificati negli immaginari
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <cutoff frequency in Hz>" << std::endl;
        return 1;
    }

    double cutoffFreq = std::stod(argv[1]);  // Frequenza di taglio letta da riga di comando

    const char* inputFile = "/home/tomas/Desktop/drums.wav";
    const char* outputFile = "/home/tomas/Desktop/drumsLPF.wav";
    SF_INFO sfInfo;
    SNDFILE* inFile = sf_open(inputFile, SFM_READ, &sfInfo);

    if (!inFile) {
        std::cerr << "[ERR] in file " << inputFile << std::endl;
        return 1;
    }

    int numSamples = sfInfo.frames * sfInfo.channels;
    
    // Alloca la memoria per i vettori real e imag, allineandoli a 16 byte
    double* real = (double*)std::aligned_alloc(16, sizeof(double) * numSamples);
    if (real == nullptr) {
        std::cerr << "[ERR] Memory allocation failed for real array!" << std::endl;
        return 1;
    }

    double* imag = (double*)std::aligned_alloc(16, sizeof(double) * numSamples);
    if (imag == nullptr) {
        std::cerr << "[ERR] Memory allocation failed for imag array!" << std::endl;
        std::free(real);  // Libera la memoria precedentemente allocata
        return 1;
    }

    std::vector<short> buffer(numSamples);

    sf_read_short(inFile, buffer.data(), numSamples);
    sf_close(inFile);

    // Copia i dati dal buffer negli array real (convertiti in double)
    for (int i = 0; i < numSamples; ++i) {
        real[i] = static_cast<double>(buffer[i]);
    }

    // Inizializza l'array imag a zero
    std::memset(imag, 0, sizeof(double) * numSamples);

    // FFT & IFFT
    fftw_complex* fftData = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * numSamples);
    fftw_complex* ifftData = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * numSamples);
    fftw_plan forwardPlan = fftw_plan_dft_r2c_1d(numSamples, real, fftData, FFTW_ESTIMATE);
    fftw_plan inversePlan = fftw_plan_dft_c2r_1d(numSamples, fftData, real, FFTW_ESTIMATE);

    // Esegui la FFT
    fftw_execute(forwardPlan);

    // Copia i dati nei vettori reali e immaginari
    for (int i = 0; i < numSamples / 2 + 1; ++i) {
        real[i] = fftData[i][0];
        imag[i] = fftData[i][1];
    }

    // Applica il filtro passa-basso
    applyLowPassFilterFFT(real, imag, numSamples / 2, cutoffFreq, SAMPLE_RATE);

    // Copia i dati filtrati per la IFFT
    for (int i = 0; i < numSamples / 2 + 1; ++i) {
        fftData[i][0] = real[i];
        fftData[i][1] = imag[i];
    }

    // Esegui la IFFT
    fftw_execute(inversePlan);

    // Normalizzazione
    double normalFactor = 1.0 / numSamples;
    for (int i = 0; i < numSamples; ++i) {
        real[i] *= normalFactor;
    }

    // Conversione del segnale a short per il salvataggio su file
    for (int i = 0; i < numSamples; ++i) {
        buffer[i] = static_cast<short>(std::round(real[i]));
    }

    // Salva il file audio filtrato
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

    // Libera le risorse
    fftw_destroy_plan(forwardPlan);
    fftw_destroy_plan(inversePlan);
    fftw_free(fftData);
    fftw_free(ifftData);

    // Libera la memoria allineata
    std::free(real);
    std::free(imag);

    return 0;
}
