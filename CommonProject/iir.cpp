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

const float LOW_GAIN = -60.0;
const float MID_GAIN = -2.0;
const float HIGH_GAIN = -8.0;
const double PI = 3.14159265358979323846;

/*  DESCRIPTION:

    3 bands Equalizer implemented with IIR filter.

    compile: 
        g++ iir.cpp -o iir -march=native
    
    run:
        ./iir
*/

int32_t gainScale(float gain){
    float pot = pow(10.0f, gain / 20.0f);
    return SCALE_FACTOR * pot;
}

// reads wav aligned
std::vector<int16_t> readWAV(const std::string& filename, int& sampleRate, int& numChannels) {
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
    int numSamples = dataSize / 2;  // 2 bytes for each sample

    // prepares array
    std::vector<int16_t> samples(numSamples);

    // reads values
    file.read(reinterpret_cast<char*>(samples.data()), dataSize);

    std::cout << "-\tNum samples: " << samples.size();

    return samples;
}

// writes a wav file
void writeWAV(const std::string& filename, const std::vector<int16_t>& samples, int sampleRate, int numChannels) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    // Header WAV standard
    char header[44] = {0};
    int32_t fileSize = 36 + samples.size() * 2;
    int32_t dataSize = samples.size() * 2;

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
    file.write(reinterpret_cast<const char*>(samples.data()), dataSize);
}

// applies eq for a single sample
void applyIIRFilterInt16SIMD(__m128i* input, __m128i* output, int numSamples, __m128i* b, __m128i* a, __m128i* z) {
    for (int i = 0; i < numSamples / 4; ++i) {
        // Load input samples and convert to 32-bit
        __m128i in = _mm_load_si128(&input[i]);
        
        // First stage computation
        __m128i temp1 = _mm_mullo_epi32(b[0], in);              // b[0] * x[n]
        __m128i temp2 = _mm_mullo_epi32(b[1], z[0]);           // b[1] * x[n-1]
        __m128i temp3 = _mm_mullo_epi32(b[2], z[1]);           // b[2] * x[n-2]
        
        // Feedback computation
        __m128i temp4 = _mm_mullo_epi32(a[1], z[0]);           // a[1] * y[n-1]
        __m128i temp5 = _mm_mullo_epi32(a[2], z[1]);           // a[2] * y[n-2]
        
        // Combine feedforward terms
        __m128i ff_sum = _mm_add_epi32(_mm_add_epi32(temp1, temp2), temp3);
        
        // Combine feedback terms
        __m128i fb_sum = _mm_add_epi32(temp4, temp5);
        
        // Final computation with scaling
        __m128i result = _mm_srai_epi32(_mm_sub_epi32(ff_sum, fb_sum), FIXED_POINT_SCALE);
        
        // Store result
        _mm_store_si128(&output[i], result);
        
        // Update delay elements
        z[1] = z[0];
        z[0] = in;
    }
}

// Sequential IIR filter
int16_t applyIIRFilterInt16Sequential(int16_t sample, int32_t* b, int32_t* a, int32_t* z) {
    int32_t input = static_cast<int32_t>(sample);
    /*
        y[n] = b[0] * x[n] + z[0]
        where:
        - y[n] is the output sample
        - x[n] is the input sample
        - b[0] is the feedforward coefficient
        - z[0] is the internal status value
        The result is divided by 65536 to downscale to 16 bit
    */ 
    int32_t output = (b[0] * input + z[0]) >> FIXED_POINT_SCALE;
    z[0] = (b[1] * input - a[1] * output + z[1]) >> FIXED_POINT_SCALE;
    z[1] = (b[2] * input - a[2] * output) >> FIXED_POINT_SCALE;
    return static_cast<int16_t>(std::min(std::max(output, -32768), 32767)); // Clamping
}

// Sequential equalizer function
uint64_t applyEqualizerSequential(std::vector<int16_t>& samples) {
    uint64_t start = __rdtsc();

    // Gains (scaled)
    int32_t lowGain = gainScale(LOW_GAIN);  // -6 dB = 0.501 scaled -> 32845
    int32_t midGain = gainScale(MID_GAIN);   // +2 dB = 1.259 scaled -> 82504
    int32_t highGain = gainScale(HIGH_GAIN); // -3 dB = 0.707 scaled -> 46395

    // Filter coefficients
    int32_t bLow[3] = {16384, 32768, 16384}; // Low-pass
    int32_t aLow[3] = {65536, -62019, 20225};
    int32_t zLow[2] = {0, 0};

    int32_t bMid[3] = {16384, 0, -16384}; // Band-pass
    int32_t aMid[3] = {65536, -62019, 20225};
    int32_t zMid[2] = {0, 0};

    int32_t bHigh[3] = {16384, -32768, 16384}; // High-pass
    int32_t aHigh[3] = {65536, -62019, 20225};
    int32_t zHigh[2] = {0, 0};

    // Apply filters
    for (auto& sample : samples) {
        int16_t low = static_cast<int16_t>((applyIIRFilterInt16Sequential(sample, bLow, aLow, zLow) * lowGain) >> 16);
        int16_t mid = static_cast<int16_t>((applyIIRFilterInt16Sequential(sample, bMid, aMid, zMid) * midGain) >> 16);
        int16_t high = static_cast<int16_t>((applyIIRFilterInt16Sequential(sample, bHigh, aHigh, zHigh) * highGain) >> 16);

        // Combine and clamp
        int32_t combined = static_cast<int32_t>(low) + static_cast<int32_t>(mid) + static_cast<int32_t>(high);
        sample = static_cast<int16_t>(std::min(std::max(combined, -32768), 32767));
    }

    return __rdtsc() - start;
}

// SIMD equalizer function (SSE2)
uint64_t applyEqualizerSIMD(std::vector<int16_t>& samples) {
    uint64_t start = __rdtsc();

    // Gains (scaled con pre-gain)
    __m128i lowGain = _mm_set1_epi32(gainScale(LOW_GAIN));
    __m128i midGain = _mm_set1_epi32(gainScale(MID_GAIN));
    __m128i highGain = _mm_set1_epi32(gainScale(HIGH_GAIN));

    // Filter coefficients (invariati)
    __m128i bLow[3] = {_mm_set1_epi32(16384), _mm_set1_epi32(32768), _mm_set1_epi32(16384)};
    __m128i aLow[3] = {_mm_set1_epi32(65536), _mm_set1_epi32(-62019), _mm_set1_epi32(20225)};
    __m128i zLow[2] = {_mm_setzero_si128(), _mm_setzero_si128()};

    __m128i bMid[3] = {_mm_set1_epi32(16384), _mm_set1_epi32(0), _mm_set1_epi32(-16384)};
    __m128i aMid[3] = {_mm_set1_epi32(65536), _mm_set1_epi32(-62019), _mm_set1_epi32(20225)};
    __m128i zMid[2] = {_mm_setzero_si128(), _mm_setzero_si128()};

    __m128i bHigh[3] = {_mm_set1_epi32(16384), _mm_set1_epi32(-32768), _mm_set1_epi32(16384)};
    __m128i aHigh[3] = {_mm_set1_epi32(65536), _mm_set1_epi32(-62019), _mm_set1_epi32(20225)};
    __m128i zHigh[2] = {_mm_setzero_si128(), _mm_setzero_si128()};

    // Process samples in chunks of 4 (SSE operates on 4 32-bit integers)
    for (size_t i = 0; i < samples.size(); i += 4) {
        // Convert 16-bit samples to 32-bit
        __m128i samples32 = _mm_cvtepi16_epi32(_mm_loadl_epi64((__m128i*)&samples[i]));

        // Process each band
        __m128i low32, mid32, high32;

        // Apply filters
        applyIIRFilterInt16SIMD(&samples32, &low32, 4, bLow, aLow, zLow);
        applyIIRFilterInt16SIMD(&samples32, &mid32, 4, bMid, aMid, zMid);
        applyIIRFilterInt16SIMD(&samples32, &high32, 4, bHigh, aHigh, zHigh);

        // Apply gains with intermediate scaling
        low32 = _mm_srai_epi32(_mm_mullo_epi32(low32, lowGain), FIXED_POINT_SCALE);
        mid32 = _mm_srai_epi32(_mm_mullo_epi32(mid32, midGain), FIXED_POINT_SCALE);
        high32 = _mm_srai_epi32(_mm_mullo_epi32(high32, highGain), FIXED_POINT_SCALE);

        // Sum all bands with intermediate clamping
        __m128i sum = _mm_add_epi32(low32, mid32);
        sum = _mm_add_epi32(sum, high32);

        // Soft clipping function (simplified tanh-like)
        // Se il valore è oltre il 75% del massimo, lo comprimiamo gradualmente
        __m128i threshold = _mm_set1_epi32(24576); // 75% di 32768
        __m128i mask = _mm_cmpgt_epi32(sum, threshold);
        __m128i diff = _mm_sub_epi32(sum, threshold);
        __m128i compressed = _mm_srai_epi32(diff, 2); // Comprimi l'eccesso di 1/4
        sum = _mm_sub_epi32(sum, _mm_and_si128(compressed, mask));

        // Negative threshold
        __m128i neg_threshold = _mm_set1_epi32(-24576);
        mask = _mm_cmplt_epi32(sum, neg_threshold);
        diff = _mm_sub_epi32(sum, neg_threshold);
        compressed = _mm_srai_epi32(diff, 2);
        sum = _mm_sub_epi32(sum, _mm_and_si128(compressed, mask));

        // Convert back to 16-bit with saturation
        __m128i result = _mm_packs_epi32(sum, sum);

        // Store result
        _mm_storel_epi64((__m128i*)&samples[i], result);
    }

    uint64_t end = __rdtsc();
    return end - start;
}


int main() {
    clock_t start;
    start = clock();

    int sampleRate, numChannels;
    std::vector<int16_t> samples = readWAV("./samples/fullSong.wav", sampleRate, numChannels);

    std::vector<int16_t> samplesSIMD = samples;
    // sequential
    uint64_t timeSequential = applyEqualizerSequential(samples);

    // parallel
    uint64_t timeSIMD = applyEqualizerSIMD(samplesSIMD);

    writeWAV("./samples/fullSongIIR.wav", samplesSIMD, sampleRate, numChannels);

    printf("\tElapsed time: %.3fs", (float) (clock() - start) / CLOCKS_PER_SEC);
    std::cout << "\tSPEEDUP " << (float)timeSequential / (float)timeSIMD << "\n" << std::endl;


    return 0;
}
