/*
    expected sum = 255 * 4_096 = 1_044_480
    8 bit sum is an overflow risk -> from 8 (max val 255) to 16 bit (max val 65_535)
    but in this case the sum is greater than 65_535 -> from 16 bit to 32 (max val about 4 billions).
*/ 

#include <stdio.h>
#include <immintrin.h>
#include <x86intrin.h>
#include <cstdint>
#include <time.h>

#define VECTOR_LENGTH 4096
#define SSE_DATA_LANE 16



unsigned long long scalarCalc(uint8_t* A){
    // time tracking
    unsigned long long startTime = __rdtsc();

    // start
    unsigned int res = 0;
    for(int i = 0; i < VECTOR_LENGTH; i++){
        res += A[i];
    }
    // end

    unsigned long long endTime = __rdtsc();
    unsigned long long duration = endTime - startTime;
    printf("\tSCALAR: sum = %u\t", res);

    return duration;
}

unsigned long long simdCalc(uint8_t* A){
    unsigned long long startTime = __rdtsc();

    // start
    __m128i sum_low = _mm_setzero_si128();  // acc array
    __m128i sum_high = _mm_setzero_si128(); // acc array
    __m128i zero = _mm_setzero_si128();
    __m128i *pA = (__m128i*) A;

    for (int i = 0; i < VECTOR_LENGTH / SSE_DATA_LANE; i++) {
        __m128i passo = _mm_load_si128(pA + i);                // load of 16 bytes
        // two pieces of size 16 managed separately, then we will "merge" them
        __m128i passo_low = _mm_unpacklo_epi8(passo, zero);    // low unpack to uint16_t
        __m128i passo_high = _mm_unpackhi_epi8(passo, zero);   // high unpack to uint16_t
        sum_low = _mm_add_epi16(sum_low, passo_low);       // low acc
        sum_high = _mm_add_epi16(sum_high, passo_high);    // high acc
    }

    // sum 32 bit : extend to 32
    __m128i sum32_low = _mm_add_epi32(_mm_unpacklo_epi16(sum_low, zero), _mm_unpackhi_epi16(sum_low, zero));
    __m128i sum32_high = _mm_add_epi32(_mm_unpacklo_epi16(sum_high, zero), _mm_unpackhi_epi16(sum_high, zero));
    // total
    __m128i total_sum = _mm_add_epi32(sum32_low, sum32_high);

    // final sum
    int tmp[4] __attribute__((aligned(SSE_DATA_LANE)));
    _mm_store_si128((__m128i*)tmp, total_sum);
    unsigned int simd_sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    // end     

    unsigned long long endTime = __rdtsc();
    unsigned long long duration = endTime - startTime;

    // print output
    printf("SIMD: sum = %u \t", simd_sum);

    return duration;
}



int main(){
    uint8_t A[VECTOR_LENGTH] __attribute__((aligned(SSE_DATA_LANE)));

    // init array
    for (int i=0; i<VECTOR_LENGTH; i++){   
        // worst case: every value is 255
        A[i]=255;
    }

    // sequential case
    unsigned long long scalarTime = scalarCalc(A);

    // SIMD case
    unsigned long long simdTime = simdCalc(A);

    float speedup = (float) scalarTime / simdTime;
    printf("SPEEDUP = %.3f\n", speedup);

    return 0;
}
