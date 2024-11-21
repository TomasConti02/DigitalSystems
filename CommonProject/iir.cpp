#include <iostream>
#include <fstream>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <immintrin.h>
#include <cstdlib>

#define FIXED_POINT_SCALE 16

/*
    3 bands Equalizer implemented with IIR filter.

    compile: 
        g++ iir.cpp -o iir -march=native
    
    run:
        ./iir
*/

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

// applies eq for a single 16-bit sample
void applyIIRFilterInt16SIMD(__m128i* input, __m128i* output, int numSamples, __m128i* b, __m128i* a, __m128i* z) {
    for (int i = 0; i < numSamples / 8; ++i) {
        // load
        __m128i in = _mm_load_si128(&input[i]);
        // execute filter
        __m128i result = _mm_srai_epi32(
            _mm_add_epi32(
                _mm_sub_epi32(
                    _mm_add_epi32(
                        _mm_mullo_epi32(b[0], in), 
                        z[0]), 
                    _mm_mullo_epi32(a[0], in)),
                z[1]),
            FIXED_POINT_SCALE);
        
        // stores result
        _mm_store_si128(&output[i], result);

        // update feedback registers
        z[0] = z[1];
        z[1] = result;
    }
}

// align vector
std::vector<int16_t> createAlignedVector(size_t size) {
    // alloc of aligned array
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 16, size * sizeof(int16_t)) != 0) {
        throw std::runtime_error("Memory allocation failed");
    }

    return std::vector<int16_t>(reinterpret_cast<int16_t*>(ptr), reinterpret_cast<int16_t*>(ptr) + size);
}

// Sequential IIR filter
int16_t applyIIRFilterInt16Sequential(int16_t sample, int32_t* b, int32_t* a, int32_t* z) {
    int32_t input = static_cast<int32_t>(sample);
    int32_t output = (b[0] * input + z[0]) >> FIXED_POINT_SCALE;
    z[0] = (b[1] * input - a[1] * output + z[1]) >> FIXED_POINT_SCALE;
    z[1] = (b[2] * input - a[2] * output) >> FIXED_POINT_SCALE;
    return static_cast<int16_t>(std::min(std::max(output, -32768), 32767)); // Clamping
}

// Sequential equalizer function
uint64_t applyEqualizerSequential(std::vector<int16_t>& samples) {
    uint64_t start = __rdtsc();

    // Gains (scaled)
    int32_t lowGain = 45875;  // -6 dB = 0.501 scaled (* 65536)
    int32_t midGain = 82207;  // +2 dB = 1.259 scaled
    int32_t highGain = 58254; // -3 dB = 0.707 scaled

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

    // Gains (scaled)
    __m128i lowGain = _mm_set1_epi32(45875);  // -6 dB = 0.501 scaled (* 65536)
    __m128i midGain = _mm_set1_epi32(82207);  // +2 dB = 1.259 scaled
    __m128i highGain = _mm_set1_epi32(58254); // -3 dB = 0.707 scaled

    // Filter coefficients
    __m128i bLow[3] = {_mm_set1_epi32(16384), _mm_set1_epi32(32768), _mm_set1_epi32(16384)}; // Low-pass
    __m128i aLow[3] = {_mm_set1_epi32(65536), _mm_set1_epi32(-62019), _mm_set1_epi32(20225)};
    __m128i zLow[2] = {_mm_setzero_si128(), _mm_setzero_si128()};

    __m128i bMid[3] = {_mm_set1_epi32(16384), _mm_set1_epi32(0), _mm_set1_epi32(-16384)}; // Band-pass
    __m128i aMid[3] = {_mm_set1_epi32(65536), _mm_set1_epi32(-62019), _mm_set1_epi32(20225)};
    __m128i zMid[2] = {_mm_setzero_si128(), _mm_setzero_si128()};

    __m128i bHigh[3] = {_mm_set1_epi32(16384), _mm_set1_epi32(-32768), _mm_set1_epi32(16384)}; // High-pass
    __m128i aHigh[3] = {_mm_set1_epi32(65536), _mm_set1_epi32(-62019), _mm_set1_epi32(20225)};
    __m128i zHigh[2] = {_mm_setzero_si128(), _mm_setzero_si128()};

    // Apply filters in SIMD mode
    size_t i = 0;
    for (; i + 8 <= samples.size(); i += 8) {
        __m128i samples16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&samples[i]));
        __m128i samples32_low = _mm_unpacklo_epi16(samples16, _mm_setzero_si128());
        __m128i samples32_high = _mm_unpackhi_epi16(samples16, _mm_setzero_si128());

        // Apply SIMD IIR filter
        applyIIRFilterInt16SIMD(&samples32_low, &samples32_low, 8, bLow, aLow, zLow);
        applyIIRFilterInt16SIMD(&samples32_high, &samples32_high, 8, bHigh, aHigh, zHigh);

        // Combine and convert back to 16-bit
        __m128i result16 = _mm_packs_epi32(samples32_low, samples32_high);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(&samples[i]), result16);
    }

    // Handle remaining samples sequentially
    for (; i < samples.size(); ++i) {
        samples[i] = applyIIRFilterInt16Sequential(samples[i], reinterpret_cast<int32_t*>(bLow), reinterpret_cast<int32_t*>(aLow), reinterpret_cast<int32_t*>(zLow));
    }

    return __rdtsc() - start;
}

int main() {
    try {
        int sampleRate, numChannels;
        std::vector<int16_t> samples = readWAV("./samples/fullSong.wav", sampleRate, numChannels);

        // sequential
        uint64_t timeSequential = applyEqualizerSequential(samples);
        std::cout << "\tSequential Time: " << timeSequential << " ticks";

        std::vector<int16_t> samplesSIMD = samples;

        // parallel
        uint64_t timeSIMD = applyEqualizerSIMD(samplesSIMD);
        std::cout << "\tSIMD Time: " << timeSIMD << " ticks" << std::endl;

        std::cout << "\n\tSPEEDUP " << (float)timeSequential / (float)timeSIMD << "\n" << std::endl;
        writeWAV("./samples/fullSongIIR.wav", samples, sampleRate, numChannels);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}
