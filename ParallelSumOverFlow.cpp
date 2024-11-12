#include <iostream>
#include <fstream>
#include <x86intrin.h>
//2^10
#define VECTOR_LENGTH 262144
#define SSE_DATA_LANE 16

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif


std::ofstream logFile("performance_log.csv");

u_int64_t sommaSSE(uint8_t* data, int length, uint32_t* risultato) {
    u_int64_t clock_counter_start = __rdtsc();
    __m128i somma = _mm_setzero_si128();
    __m128i passo, passo2, passo3;
    __m128i* p_DATA = (__m128i*) data;
    //((2^16-1)/255)/2=127.5 =>127
    for (int i = 0; i < 128; i++) {
        passo = _mm_load_si128(p_DATA + i);
        passo2 = _mm_cvtepu8_epi16(passo);  // Conversione unsigned
        passo3 = _mm_cvtepu8_epi16(_mm_srli_si128(passo, 8));
        somma = _mm_add_epi16(_mm_add_epi16(passo2, passo3), somma);
    }

    passo2 = _mm_cvtepu16_epi32(somma);  // Conversione unsigned
    passo3 = _mm_cvtepu16_epi32(_mm_srli_si128(somma, 8));
    somma = _mm_add_epi32(passo2, passo3);

    for (int i = 128; i < (length / SSE_DATA_LANE); i++) {
        passo = _mm_load_si128(p_DATA + i);
        __m128i passo1 = _mm_cvtepu8_epi32(passo);  // Conversione unsigned
        passo2 = _mm_cvtepu8_epi32(_mm_srli_si128(passo, 4));
        passo3 = _mm_cvtepu8_epi32(_mm_srli_si128(passo, 8));
        __m128i passo4 = _mm_cvtepu8_epi32(_mm_srli_si128(passo, 12));
        somma = _mm_add_epi32(_mm_add_epi32(passo1, _mm_add_epi32(passo2, _mm_add_epi32(passo3, passo4))), somma);
    }

    somma = _mm_hadd_epi32(somma, somma);
    somma = _mm_hadd_epi32(somma, somma);
    *risultato = _mm_cvtsi128_si32(somma); //estraggo un singolo valore da 32bit dal registro 

    u_int64_t clock_counter_end = __rdtsc();
    logFile << "SSE, " << clock_counter_end - clock_counter_start << "\n";
    return (clock_counter_end - clock_counter_start);
}


u_int64_t sommaSequenziale(uint8_t* data, int length, uint32_t* risultato) {
    u_int64_t clock_counter_start = __rdtsc();
    uint32_t somma = 0;

    for (int i = 0; i < length; i++) {
        somma += data[i];
    }

    *risultato = somma;
    u_int64_t clock_counter_end = __rdtsc();
    logFile << "Sequenziale, " << clock_counter_end - clock_counter_start << "\n";
    return (clock_counter_end - clock_counter_start);
}

int main() {
    uint8_t data[VECTOR_LENGTH];
    for (int i = 0; i < VECTOR_LENGTH; ++i) {
        data[i] = 255;  // Riempie l'array con valori ripetuti (255)
    }

    double totaleSpeedup = 0.0;  // Variabile per accumulare il totale degli speedup

    for (int iterazione = 0; iterazione < 20; ++iterazione) {
        uint32_t risultatoSSE = 0;
        uint32_t risultatoSeq = 0;

        u_int64_t tempoSSE = sommaSSE(data, VECTOR_LENGTH, &risultatoSSE);
        u_int64_t tempoSeq = sommaSequenziale(data, VECTOR_LENGTH, &risultatoSeq);

        if (risultatoSSE == risultatoSeq) {
            double speedup = static_cast<double>(tempoSeq) / tempoSSE;
            totaleSpeedup += speedup;  // Aggiungi lo speedup alla somma totale

            std::cout << "Iterazione " << iterazione + 1 << ":\n";
            std::cout << "  Risultati uguali.\n";
            std::cout << "  Tempo SSE: " << tempoSSE << " cicli di clock\n";
            std::cout << "  Tempo Sequenziale: " << tempoSeq << " cicli di clock\n";
            std::cout << "  Speed-up: " << speedup << "\n";

            // Log dell'indice di iterazione e dello speed-up
            logFile << "Iterazione " << iterazione + 1 << ", Speed-up: " << speedup << "\n";
        } else {
            std::cout << "Errore: i risultati non coincidono!\n";
            logFile << "Errore alla iterazione " << iterazione + 1 << "\n";
        }
    }

    // Calcola e stampa la media dello speedup
    double mediaSpeedup = totaleSpeedup / 20.0;
    std::cout << "Media dello Speed-up: " << mediaSpeedup << "\n";
    logFile << "Media dello Speed-up: " << mediaSpeedup << "\n";

    logFile.close();
    return 0;
}
