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

#define SAMPLE_RATE 44100  // Sample rate (Hz)

// function to apply the equalization
void applyEqualizer(double* real, double* imag, int numSamples, int sampleRate) {
    u_int64_t clock_counter_start = __rdtsc();
    int lowBandEnd = static_cast<int>(300 / (static_cast<double>(sampleRate) / numSamples)); // up to 300 Hz
    int midBandStart = lowBandEnd;
    int midBandEnd = static_cast<int>(3000 / (static_cast<double>(sampleRate) / numSamples)); // from 300 to 3000 Hz
    int highBandStart = midBandEnd;

    // uses the formula gain = 20log(in/out) in reverse to calculate the multiplication factor for the convolution
    double lowGain = std::pow(10, -6.0 / 20.0);    // -6 dB
    double midGain = std::pow(10, 2.0 / 20.0);     // +2 dB
    double highGain = std::pow(10, -3.0 / 20.0);   // -3 dB

    // low frequencies
    for (int i = 0; i < lowBandEnd; ++i) {
        real[i] *= lowGain;
        imag[i] *= lowGain;
    }

    // mid frequencies
    for (int i = midBandStart; i < midBandEnd; ++i) {
        real[i] *= midGain;
        imag[i] *= midGain;
    }

    // high frequencies
    for (int i = highBandStart; i < numSamples / 2; ++i) {
        real[i] *= highGain;
        imag[i] *= highGain;
    }
    u_int64_t clock_counter_end = __rdtsc();
    printf("Time: %li\n", clock_counter_end - clock_counter_start);
}

int main(int argc, char* argv[]) {
    const char* inputFile = "./samples/drums.wav";
    const char* outputFile = "./samples/drumsEQ.wav";
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
    applyEqualizer(real.data(), imag.data(), numSamples / 2, SAMPLE_RATE);

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
