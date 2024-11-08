#include <iostream>
#include <vector>
#include <cmath>
#include <sndfile.h>
#include <fftw3.h>

/*

    This is a test to start with audio processing in C++.
    This program applies a Low-Pass Filter on a pre-specified wav file, with the cutoff frequency specified as a parameter.

    --- HOW TO RUN ---

    There are two libraries required. 
    Type on CLI:
    
    sudo apt-get install libsndfile1-dev
    sudo apt-get install libfftw3-dev
    g++ -o scalarLPF scalarLPF.cpp -lsndfile -lfftw3 -lm
    ./scalarLPF 1000

    The last parameter is the cutoff frequency in Hz.

*/

#define SAMPLE_RATE 44100  // Sample rate (Hz)

// Real and imaginary parts are stored in two separate arrays
void applyLowPassFilterFFT(double* real, double* imag, int numSamples, double cutoffFreq, int sampleRate) {
    int cutoffIndex = static_cast<int>(cutoffFreq / (static_cast<double>(sampleRate) / numSamples));
    for (int i = cutoffIndex; i < numSamples - cutoffIndex; ++i) {
        real[i] = 0.0;
        imag[i] = 0.0;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <cutoff frequency in Hz>" << std::endl;
        return 1;
    }

    double cutoffFreq = std::stod(argv[1]);  // Frequenza di taglio letta da riga di comando

    const char* inputFile = "./samples/drums.wav";
    const char* outputFile = "drumsLPF.wav";
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

    // Esegue la FFT
    fftw_execute(forwardPlan);

    // Copia i dati nei vettori reali e immaginari
    for (int i = 0; i < numSamples / 2 + 1; ++i) {
        real[i] = fftData[i][0];
        imag[i] = fftData[i][1];
    }

    // Applica il filtro passa-basso
    applyLowPassFilterFFT(real.data(), imag.data(), numSamples / 2, cutoffFreq, SAMPLE_RATE);

    // Copia i dati filtrati per la IFFT
    for (int i = 0; i < numSamples / 2 + 1; ++i) {
        fftData[i][0] = real[i];
        fftData[i][1] = imag[i];
    }

    // Esegue la IFFT
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

    return 0;
}
