#include <iostream>
#include <fstream>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <immintrin.h>
#include <cstdlib>
#include <cmath>
#include <time.h>

#define FIXED_POINT_SCALE 16
#define SCALE_FACTOR 65536

const float LOW_GAIN = -3.0;
const float MID_GAIN = 0.0;
const float HIGH_GAIN = -2.0;

/*  DESCRIPTION:

    3 bands Equalizer implemented with IIR filter.

    compile: 
        g++ iir.cpp -o iir -march=native
    
    run:
        ./iir
*/

/*
    Takes the gain in dB and uses the formula:
    g = 20log(R) to compute  R
*/
int32_t gainScale(float gain) {
// Special case for no gain
if (std::abs(gain) < 0.001f) return SCALE_FACTOR;
float linearGain = powf(10.0f, gain / 20.0f);
return static_cast<int32_t>(SCALE_FACTOR * linearGain);
}

// reads wav aligned
int16_t* readWAV(const std::string& filename, int& sampleRate, int& numChannels, int& numSamples) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    // reads header of wav file
    char header[44];
    file.read(header, 44); // 44 byte header

    sampleRate = *reinterpret_cast<int32_t*>(&header[24]);  // sample rate
    numChannels = *reinterpret_cast<int16_t*>(&header[22]); // number of channels

    // computes number of samples
    int32_t dataSize = *reinterpret_cast<int32_t*>(&header[40]);
    numSamples = dataSize / 2;  // 2 bytes for each sample

    // Allocate aligned memory
    int16_t* samples = reinterpret_cast<int16_t*>(_mm_malloc(numSamples * sizeof(int16_t), 16));
    
    // reads values
    file.read(reinterpret_cast<char*>(samples), dataSize);

    std::cout << "-\tNsamples: " << numSamples;

    return samples;
}

// writes a wav file
void writeWAV(const std::string& filename, int16_t* samples, int numSamples, int sampleRate, int numChannels) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    // Header WAV standard
    char header[44] = {0};
    int32_t fileSize = 36 + numSamples * 2;
    int32_t dataSize = numSamples * 2;

    std::copy("RIFF", "RIFF" + 4, header);
    *reinterpret_cast<int32_t*>(&header[4]) = fileSize;
    std::copy("WAVE", "WAVE" + 4, &header[8]);
    std::copy("fmt ", "fmt " + 4, &header[12]);
    *reinterpret_cast<int32_t*>(&header[16]) = 16;
    *reinterpret_cast<int16_t*>(&header[20]) = 1;
    *reinterpret_cast<int16_t*>(&header[22]) = numChannels;
    *reinterpret_cast<int32_t*>(&header[24]) = sampleRate;
    *reinterpret_cast<int32_t*>(&header[28]) = sampleRate * numChannels * 2;
    *reinterpret_cast<int16_t*>(&header[32]) = numChannels * 2;
    *reinterpret_cast<int16_t*>(&header[34]) = 16;
    std::copy("data", "data" + 4, &header[36]);
    *reinterpret_cast<int32_t*>(&header[40]) = dataSize;

    file.write(header, 44);
    file.write(reinterpret_cast<const char*>(samples), dataSize);
}

// Sequential IIR filter
int16_t applyIIRFilterSequential(int16_t sample, int32_t* b, int32_t* a, int32_t* z) {
    int32_t input = static_cast<int32_t>(sample);
    int32_t output = (b[0] * input + z[0]) >> FIXED_POINT_SCALE;
    z[0] = (b[1] * input - a[1] * output + z[1]) >> FIXED_POINT_SCALE;
    z[1] = (b[2] * input - a[2] * output) >> FIXED_POINT_SCALE;
    return static_cast<int16_t>(std::min(std::max(output, -32768), 32767)); // Clamping
}

// Sequential equalizer function
uint64_t applyEqualizerSequential(int16_t* samples, int numSamples) {
    uint64_t start = __rdtsc();
    
    // Gains (scaled)
    int32_t lowGain = gainScale(LOW_GAIN);   
    int32_t midGain = gainScale(MID_GAIN);   
    int32_t highGain = gainScale(HIGH_GAIN);
    
    // Filter coefficients
    int32_t bLow[3] = {16384, 32768, 16384};  // Low-pass
    int32_t aLow[3] = {65536, -62019, 20225};
    int32_t zLow[2] = {0, 0};
    
    int32_t bMid[3] = {16384, 0, -16384};     // Band-pass
    int32_t aMid[3] = {65536, -62019, 20225};
    int32_t zMid[2] = {0, 0};
    
    int32_t bHigh[3] = {16384, -32768, 16384}; // High-pass
    int32_t aHigh[3] = {65536, -62019, 20225};
    int32_t zHigh[2] = {0, 0};
    
    // Apply filters with gain
    for (int i = 0; i < numSamples; i++) {
        // First, separate bands by filtering
        int16_t low = applyIIRFilterSequential(samples[i], bLow, aLow, zLow);
        int16_t mid = applyIIRFilterSequential(samples[i], bMid, aMid, zMid);
        int16_t high = applyIIRFilterSequential(samples[i], bHigh, aHigh, zHigh);
        
        // Then apply gains to each separated band
        low = static_cast<int16_t>((static_cast<int32_t>(low) * lowGain) >> 16);
        mid = static_cast<int16_t>((static_cast<int32_t>(mid) * midGain) >> 16);
        high = static_cast<int16_t>((static_cast<int32_t>(high) * highGain) >> 16);
        
        // Combine and clamp
        int32_t combined = static_cast<int32_t>(low) + static_cast<int32_t>(mid) + static_cast<int32_t>(high);
        samples[i] = static_cast<int16_t>(std::min(std::max(combined, -32768), 32767));
    }
    
    return __rdtsc() - start;
}

void applyIIRFilterSIMD(__m128i* input, __m128i* output, int numSamples, __m128i* b, __m128i* a, __m128i* z) {
    __m128i in = _mm_load_si128(input);

    // feedforward
    __m128i temp1 = _mm_mullo_epi32(b[0], in);        // b[0] * x[n]
    __m128i temp2 = _mm_mullo_epi32(b[1], z[0]);      // b[1] * x[n-1]
    __m128i temp3 = _mm_mullo_epi32(b[2], z[1]);      // b[2] * x[n-2]
    __m128i ff_sum = _mm_add_epi32(_mm_add_epi32(temp1, temp2), temp3); // ff = b0*x + b1*z0 + b2*z1

    // feedback
    __m128i temp4 = _mm_mullo_epi32(a[1], z[0]);      // a[1] * y[n-1]
    __m128i temp5 = _mm_mullo_epi32(a[2], z[1]);      // a[2] * y[n-2]
    __m128i fb_sum = _mm_add_epi32(temp4, temp5);     // fb = a1*z0 + a2*z1

    __m128i result = _mm_srai_epi32(_mm_sub_epi32(ff_sum, fb_sum), FIXED_POINT_SCALE); // ff - fb

    // Clamping
    result = _mm_max_epi16(result, _mm_set1_epi16(-32768));
    result = _mm_min_epi16(result, _mm_set1_epi16(32767));

    // update status
    z[1] = z[0];
    z[0] = result;

    // save output
    _mm_store_si128(output, result);
}

// SIMD equalizer function (SSE2)
uint64_t applyEqualizerSIMD(int16_t* samples, int numSamples) {
    uint64_t start = __rdtsc();

    // Gains
    __m128i lowGain = _mm_set1_epi32(gainScale(LOW_GAIN));
    __m128i midGain = _mm_set1_epi32(gainScale(MID_GAIN));
    __m128i highGain = _mm_set1_epi32(gainScale(HIGH_GAIN));

    // Filter coefficients a, b, z
    __m128i bLow[3] = {_mm_set1_epi32(16384), _mm_set1_epi32(32768), _mm_set1_epi32(16384)};
    __m128i aLow[3] = {_mm_set1_epi32(65536), _mm_set1_epi32(-62019), _mm_set1_epi32(20225)};
    __m128i zLow[2] = {_mm_setzero_si128(), _mm_setzero_si128()};

    __m128i bMid[3] = {_mm_set1_epi32(16384), _mm_set1_epi32(0), _mm_set1_epi32(-16384)};
    __m128i aMid[3] = {_mm_set1_epi32(65536), _mm_set1_epi32(-62019), _mm_set1_epi32(20225)};
    __m128i zMid[2] = {_mm_setzero_si128(), _mm_setzero_si128()};

    __m128i bHigh[3] = {_mm_set1_epi32(16384), _mm_set1_epi32(-32768), _mm_set1_epi32(16384)};
    __m128i aHigh[3] = {_mm_set1_epi32(65536), _mm_set1_epi32(-62019), _mm_set1_epi32(20225)};
    __m128i zHigh[2] = {_mm_setzero_si128(), _mm_setzero_si128()};

    // Process samples in chunks of 8 (SSE operates on 4 32-bit integers)
    for (int i = 0; i < numSamples; i += 8) {
        // Convert 16-bit samples to 32-bit (cvt = SSE 4.1)
        __m128i samples1_32 = _mm_cvtepi16_epi32(_mm_loadl_epi64((__m128i*)&samples[i]));
        __m128i samples2_32 = _mm_cvtepi16_epi32(_mm_loadl_epi64((__m128i*)&samples[i+4]));

        // Process each band
        __m128i low1_32, mid1_32, high1_32;
        __m128i low2_32, mid2_32, high2_32;

        // Apply filters
        applyIIRFilterSIMD(&samples1_32, &low1_32, 4, bLow, aLow, zLow);
        applyIIRFilterSIMD(&samples1_32, &mid1_32, 4, bMid, aMid, zMid);
        applyIIRFilterSIMD(&samples1_32, &high1_32, 4, bHigh, aHigh, zHigh);

        applyIIRFilterSIMD(&samples2_32, &low2_32, 4, bLow, aLow, zLow);
        applyIIRFilterSIMD(&samples2_32, &mid2_32, 4, bMid, aMid, zMid);
        applyIIRFilterSIMD(&samples2_32, &high2_32, 4, bHigh, aHigh, zHigh);

        // Apply gains with intermediate scaling
        low1_32 = _mm_srai_epi32(_mm_mullo_epi32(low1_32, lowGain), FIXED_POINT_SCALE);
        mid1_32 = _mm_srai_epi32(_mm_mullo_epi32(mid1_32, midGain), FIXED_POINT_SCALE);
        high1_32 = _mm_srai_epi32(_mm_mullo_epi32(high1_32, highGain), FIXED_POINT_SCALE);

        low2_32 = _mm_srai_epi32(_mm_mullo_epi32(low2_32, lowGain), FIXED_POINT_SCALE);
        mid2_32 = _mm_srai_epi32(_mm_mullo_epi32(mid2_32, midGain), FIXED_POINT_SCALE);
        high2_32 = _mm_srai_epi32(_mm_mullo_epi32(high2_32, highGain), FIXED_POINT_SCALE);

        // Sum all bands with intermediate clamping: low + mid + high
        __m128i sum1 = _mm_add_epi32(low1_32, mid1_32);
        sum1 = _mm_add_epi32(sum1, high1_32);

        __m128i sum2 = _mm_add_epi32(low2_32, mid2_32);
        sum2 = _mm_add_epi32(sum2, high2_32);

        // Convert back to 16-bit with saturation
        __m128i result1 = _mm_packs_epi32(sum1, sum1);
        __m128i result2 = _mm_packs_epi32(sum2, sum2);

        // Store results
        _mm_storel_epi64((__m128i*)&samples[i], result1);
        _mm_storel_epi64((__m128i*)&samples[i+4], result2);
    }

    uint64_t end = __rdtsc();
    return end - start;
}

int main() {
    clock_t start;
    start = clock();
    int sampleRate, numChannels, numSamples;
    int16_t* samples = readWAV("./samples/fullSong.wav", sampleRate, numChannels, numSamples);
    
    // Create an aligned copy for sequential processing
    int16_t* samplesSequential = reinterpret_cast<int16_t*>(_mm_malloc(numSamples * sizeof(int16_t), 16));
    std::copy(samples, samples + numSamples, samplesSequential);
    
    // Process samples
    uint64_t timeSequential = applyEqualizerSequential(samplesSequential, numSamples);
    uint64_t timeSIMD = applyEqualizerSIMD(samples, numSamples);
    writeWAV("./samples/fullSongIIR.wav", samples, numSamples, sampleRate, numChannels); 
    printf("\tElapsed time: %.3fs", (float) (clock() - start) / CLOCKS_PER_SEC);
    std::cout << "\tSPEEDUP " << (float)timeSequential / (float)timeSIMD << "\n" << std::endl;
    // Free aligned memory
    _mm_free(samples);
    _mm_free(samplesSequential);

    return 0;
}
