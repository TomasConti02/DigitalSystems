//g++ -mavx2 -o programma_nome ArraySunSIMD.cpp 
#include <stdio.h>
#include <stdint.h>
#include <immintrin.h> // Intrinseci di Intel x86
#include <x86intrin.h>  // Include per le istruzioni SSE
#include <iostream>     // Necessario per std::cout e std::endl

#define VECTOR_LENGTH 262144 // Array statico di 16 elementi (multiplo di 16)
#define SSE_DATA_LANE 16   // 16 byte -> 128 bit di una linea
#define DATA_SIZE 1

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

void print_register16(__m128i reg) {
    // Array temporaneo per memorizzare il contenuto del registro
    uint16_t temp[8] __attribute__((aligned(SSE_DATA_LANE)));

    // Memorizza il contenuto del registro nell'array temporaneo
    _mm_store_si128((__m128i*)temp, reg);

    // Stampa il contenuto dell'array
    printf("Contenuto del registro 16bit: ");
    for (int i = 0; i < 8; i++) {
        printf("%d ", temp[i]);
    }
    printf("\n");
}
void print_register32(__m128i reg) {
    // Array temporaneo per memorizzare il contenuto del registro
    uint32_t temp[4] __attribute__((aligned(SSE_DATA_LANE)));

    // Memorizza il contenuto del registro nell'array temporaneo
    _mm_store_si128((__m128i*)temp, reg);

    // Stampa il contenuto dell'array
    printf("Contenuto del registro 32bit: ");
    for (int i = 0; i < 4; i++) {
        printf("%d ", temp[i]);
    }
    printf("\n");
}
void print_register8(__m128i reg) {
    // Array temporaneo per memorizzare il contenuto del registro
    uint8_t temp[16] __attribute__((aligned(SSE_DATA_LANE)));

    // Memorizza il contenuto del registro nell'array temporaneo
    _mm_store_si128((__m128i*)temp, reg);

    // Stampa il contenuto dell'array
    printf("Contenuto del registro 8bit : ");
    for (int i = 0; i < 16; i++) {
        printf("%d ", temp[i]);
    }
    printf("\n");
}
// Funzione per stampare il contenuto di DATA in righe ordinate di 16 valori ciascuna
void printDataOrdinato(uint8_t* data, int length) {
    printf("Contenuto di DATA:\n");
    for (int i = 0; i < length; i++) {
        printf("%3d ", data[i]);
        if ((i + 1) % 16 == 0) {  // Stampa una nuova riga ogni 16 valori
            printf("\n");
        }
    }
    // Aggiunge una nuova riga se l'array non è multiplo esatto di 16
    if (length % 16 != 0) {
        printf("\n");
    }
}
int main() {
    uint8_t DATA[VECTOR_LENGTH] __attribute__((aligned(SSE_DATA_LANE))); 
    for (int i = 0; i < VECTOR_LENGTH; i++) {
        DATA[i] = 1;
    }
    //printDataOrdinato(DATA, VECTOR_LENGTH);
    __m128i passo;
    __m128i passo2;
    __m128i passo3;
   __m128i  passo4;
   __m128i  passo1;
    __m128i somma = _mm_setzero_si128(); 
    __m128i zero = _mm_setzero_si128(); // XMM_0 con valori tutti a 0
    __m128i *p_DATA = (__m128i*) DATA;
      int i;
    //PRIMA OPZIONE
    printf("\nPRIMA OPZIONE \n ");
        printf("\nsiamo nel caso di 16bit\n");
        for (i = 0; i < (VECTOR_LENGTH / SSE_DATA_LANE); i++) {
            passo = _mm_load_si128(p_DATA + i);
            //passo2 = _mm_unpacklo_epi8(passo, zero); // Unpack low
            //passo3 = _mm_unpackhi_epi8(passo, zero); // Unpack high
            //print_register16(passo2);
           // print_register16(passo3);
            passo2= _mm_cvtepu8_epi16(passo);
            passo3=_mm_cvtepu8_epi16(_mm_srli_si128(passo, 8));
            somma = _mm_add_epi16(_mm_add_epi16(passo2, passo3), somma);
        }
        printf("\nrisulatato con 16 -> ");
        print_register16(somma);
        somma = _mm_setzero_si128(); 
        printf("\nsiamo nel caso di 32bit\n");
        for (i = 0; i < (VECTOR_LENGTH / SSE_DATA_LANE); i++) {
            passo = _mm_load_si128(p_DATA + i); //leggo 16 valori a 8 bit per volta 
            //passo2 = _mm_unpacklo_epi16(passo, zero); // Unpack low
            //passo3 = _mm_unpackhi_epi16(passo, zero); // Unpack high
            passo1= _mm_cvtepu8_epi32(passo);
            passo2=_mm_cvtepu8_epi32(_mm_srli_si128(passo, 4));
            passo3=_mm_cvtepu8_epi32(_mm_srli_si128(passo, 4));
            passo4=_mm_cvtepu8_epi32(_mm_srli_si128(passo, 4));
            //print_register16(passo2);
           // print_register16(passo3);
            somma = _mm_add_epi32(_mm_add_epi32(passo1, _mm_add_epi32(passo2, _mm_add_epi32(passo3, passo4))), somma);
        }
        printf("\nrisulatato con 32 -> ");
        print_register32(somma);
    printf("\nSECONDA OPZIONE \n");
       somma = _mm_setzero_si128(); 
    //SECONDA OPZIONE
    //sommiamo valori di dimensione massima 8bit in registri a 16bit(memorizzo 8 valori)
    //2^16 = 65536, il massimo dato che posso sommare è 256 => 65536/256=256 cicli prima di passare a registri a 32bit
        for (i = 0; i < 256; i++) {
            passo = _mm_load_si128(p_DATA + i);
            passo2= _mm_cvtepu8_epi16(passo);
            passo3=_mm_cvtepu8_epi16(_mm_srli_si128(passo, 8));
            somma = _mm_add_epi16(_mm_add_epi16(passo2, passo3), somma);
        }
        print_register16(somma);
        //cambio registro, passiamo da 16bit -> 32bit, abbiamo passato i 256 cicli. Evitiamo eventuali overflow
        passo2= _mm_cvtepu16_epi32(somma);
        passo3=_mm_cvtepu16_epi32(_mm_srli_si128(somma, 8));
        somma=_mm_add_epi32(passo2, passo3);
        //2^32=4294967296  =>4294967296/256=16777216 il numero di cicli massimi prima che ci sia la possibilità di over flow
        for (i = 256; i < (VECTOR_LENGTH / SSE_DATA_LANE); i++) {
            passo = _mm_load_si128(p_DATA + i);
            passo1= _mm_cvtepu8_epi32(passo);
            passo2=_mm_cvtepu8_epi32(_mm_srli_si128(passo, 4));
            passo3=_mm_cvtepu8_epi32(_mm_srli_si128(passo, 4));
            passo4=_mm_cvtepu8_epi32(_mm_srli_si128(passo, 4));
            somma = _mm_add_epi32(_mm_add_epi32(passo1, _mm_add_epi32(passo2, _mm_add_epi32(passo3, passo4))), somma);
        }
        print_register32(somma);
    return 0;
}
/*else if(i<16777216){
             passo2 = _mm_unpacklo_epi16(zero, passo); // Unpack low
             passo3 = _mm_unpacklo_epi16(zero, passo); // Unpack high
             time = _mm_add_epi32(passo2, passo3); // Unpack high
        }*/
        /*
 #include <immintrin.h>
#include <stdio.h>
#include <stdint.h>

void print_register_16(__m128i reg) {
    // Array temporaneo per memorizzare il contenuto del registro
    uint16_t temp[8] __attribute__((aligned(16)));

    // Memorizza il contenuto del registro nell'array temporaneo
    _mm_store_si128((__m128i*)temp, reg);

    // Stampa il contenuto dell'array
    printf("Contenuto del registro (16 bit): ");
    for (int i = 0; i < 8; i++) {
        printf("%u ", temp[i]);
    }
    printf("\n");
}

int main() {
    // Registro b contenente valori a 8 bit senza segno
    __m128i b = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

    // Registro a contenente tutti zeri
    __m128i a = _mm_setzero_si128();

    // Estrai i primi 8 valori di b, espandendoli a 16 bit senza segno
    __m128i low_half = _mm_cvtepu8_epi16(b);
    print_register_16(low_half);

    // Shift per ottenere l'altra metà
    __m128i high_half = _mm_cvtepu8_epi16(_mm_srli_si128(b, 8));
    print_register_16(high_half);

    return 0;
}
*/
/*#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>

void print_register_32(__m128i reg) {
    // Array temporaneo per memorizzare il contenuto del registro
    uint32_t temp[4] __attribute__((aligned(16)));

    // Memorizza il contenuto del registro nell'array temporaneo
    _mm_store_si128((__m128i*)temp, reg);

    // Stampa il contenuto dell'array
    printf("Contenuto del registro (32 bit): ");
    for (int i = 0; i < 4; i++) {
        printf("%u ", temp[i]);
    }
    printf("\n");
}

int main() {
    // Registro contenente valori a 16 bit senza segno
    __m128i a = _mm_setr_epi16(0, 100, 200, 300, 400, 500, 600, 700);

    // Prende la prima metà e la converte in 32 bit senza segno
    __m128i low_half_32 = _mm_cvtepu16_epi32(a);
    print_register_32(low_half_32);

    // Shift per ottenere l'altra metà e convertire anche quella in 32 bit senza segno
    __m128i high_half_32 = _mm_cvtepu16_epi32(_mm_srli_si128(a, 8));
    print_register_32(high_half_32);

    return 0;
}

#include <emmintrin.h>
#include <iostream>

void print_register_32(__m128i reg) {
    int* data = (int*)&reg;
    for (int i = 0; i < 4; ++i) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    // Registro contenente 8 valori a 16 bit
    __m128i b = _mm_setr_epi16(224, 232, 240, 248, 256, 264, 272, 280);

    // Estrai i primi 4 valori a 16 bit, espandendoli a 32 bit
    __m128i low_half = _mm_cvtepu16_epi32(b);
    print_register_32(low_half);

    // Estrai i successivi 4 valori a 16 bit, espandendoli a 32 bit, //preceduto da uno shit di 8byte
    __m128i high_half = _mm_cvtepu16_epi32(_mm_srli_si128(b, 8));
    print_register_32(high_half);

    return 0;
}*/
