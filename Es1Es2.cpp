#include <stdio.h>
#include <stdint.h>
#include <immintrin.h> // Intrinseci di Intel x86

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#define VECTOR_LENGTH 256 // Array statico di 16 elementi (multiplo di 16)
#define SSE_DATA_LANE 16 // 16 byte -> 128 bit di una linea
#define DATA_SIZE 1
void print_register(__m128i reg)
{
    // Array temporaneo per memorizzare il contenuto del registro
    int8_t temp[16] __attribute__((aligned(SSE_DATA_LANE)));

    // Memorizza il contenuto del registro nell'array temporaneo
    _mm_store_si128((__m128i*)temp, reg);

    // Stampa il contenuto dell'array
    printf("Contenuto del registro: ");
    for (int i = 0; i < 16; i++)
    {
        printf("%d ", temp[i]);
    }
    printf("\n");
}

void print_output(int8_t *A, int length)
{
    for (int i = 0; i < length; i++)
    {
        printf("A[%d] = %d\n", i, A[i]);
    }
}
void print_bits(int8_t value) {
    for (int i = 7; i >= 0; i--) { // Stampa i bit da MSB a LSB
        printf("%d", (value >> i) & 1);
    }
}
void print_register_bit(__m128i reg) {
    // Array temporaneo per memorizzare il contenuto del registro
    int8_t temp[VECTOR_LENGTH] __attribute__((aligned(SSE_DATA_LANE)));

    // Memorizza il contenuto del registro nell'array temporaneo
    _mm_store_si128((__m128i*)temp, reg);

    // Stampa il contenuto dell'array e i bit corrispondenti
    printf("Contenuto del registro: ");
    for (int i = 0; i < VECTOR_LENGTH; i++) {
        printf("%d (", temp[i]);
        print_bits(temp[i]); // Stampa i bit del valore
        printf(") ");
    }
    printf("\n");
}
int main()
{
    //ESERCIZIO 1 
    /*
    int8_t DATA[VECTOR_LENGTH] __attribute__((aligned(SSE_DATA_LANE))); 
    int8_t TIME[16] __attribute__((aligned(SSE_DATA_LANE))); 
    for (int i = 0; i < VECTOR_LENGTH; i++)
    {
        DATA[i] = 0;
    }
    DATA[4] = -58;
    DATA[0]=-1;
    DATA[30]=-60;
    DATA[50]=-61;
    DATA[126]=-62;
    DATA[127]=-20;
    DATA[VECTOR_LENGTH-1]=-70;
    printf("\n");
    __m128i *p_DATA = (__m128i*) DATA;
    __m128i min = _mm_set1_epi8(127);
    __m128i passo;
    u_int64_t clock_counter_start = __rdtsc();
    for(int i=0; i<(VECTOR_LENGTH/SSE_DATA_LANE/DATA_SIZE); i++){
        passo=_mm_load_si128(p_DATA+i);
        min=_mm_min_epi8(min, passo);
    }
    print_register(min);
    _mm_store_si128((__m128i*)TIME, min);
    int8_t min_value=TIME[0];
    for (int i = 1; i < 16; i++) {
        if (TIME[i] < min_value) {
            min_value = TIME[i];
        }
    }
    u_int64_t clock_counter_end = __rdtsc();
    printf("\nil min : %d\n", min_value);
    printf("\nElapsed clocks: %lu\n", clock_counter_end-clock_counter_start);*/
    //ESERCIZIO 2
    //Probelma : indici sono registri a 8bit con segno 
   int8_t DATA[VECTOR_LENGTH] __attribute__((aligned(SSE_DATA_LANE)));
    uint8_t INDICI[VECTOR_LENGTH] __attribute__((aligned(SSE_DATA_LANE)));
    int8_t TIME[16] __attribute__((aligned(SSE_DATA_LANE)));
    uint8_t TIME2[16] __attribute__((aligned(SSE_DATA_LANE)));

    // Inizializza dati e indici (ripetizione modulo 256 per rimanere nell'intervallo 0-255)
    for (int i = 0; i < VECTOR_LENGTH; i++) {
        DATA[i] = 0;
        INDICI[i] = i ; // Assicura che gli indici siano tra 0 e 255
    }

    // Imposta valori di test
    DATA[4] = -58;
    DATA[0] = -1;
    DATA[50] = -100;
    DATA[19] = -90;
    DATA[241] = -120;
    DATA[VECTOR_LENGTH - 1] = -70;

    __m128i *p_DATA = (__m128i*) DATA;
    __m128i *p_INDICI = (__m128i*) INDICI;
    __m128i min = _mm_set1_epi8(127);
    __m128i indici = _mm_set1_epi8(0);

    __m128i passo, mintime, passoIndici;
    u_int64_t clock_counter_start = __rdtsc();

    for (int i = 0; i < (VECTOR_LENGTH / SSE_DATA_LANE); i++) {
        passoIndici = _mm_load_si128(p_INDICI + i);
        passo = _mm_load_si128(p_DATA + i);
        mintime = min;
        min = _mm_min_epi8(min, passo);
        
        // Aggiorna indici solo se min è stato aggiornato
        indici = _mm_blendv_epi8(passoIndici, indici, _mm_cmpeq_epi8(min, mintime));
    }
    // Conserva il risultato
    _mm_store_si128((__m128i*)TIME, min);
    _mm_store_si128((__m128i*)TIME2, indici);
    int8_t min_value = TIME[0];
    uint8_t indiceMin = TIME2[0];
    for (int i = 1; i < 16; i++) {
        if (TIME[i] < min_value) {
            min_value = TIME[i];
            indiceMin = TIME2[i];
        }
    }
    u_int64_t clock_counter_end = __rdtsc();
    printf("\nIl minimo: %d ha indice relativo: %u\n", min_value, indiceMin);
    printf("\nCicli di clock trascorsi: %lu\n", clock_counter_end - clock_counter_start);

    return 0;
}
