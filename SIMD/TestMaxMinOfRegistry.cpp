#include <stdio.h>
#include <stdint.h>
#include <immintrin.h> // Intrinseci di Intel x86

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#define VECTOR_LENGTH 16 // Array statico di 16 elementi (multiplo di 16)
#define SSE_DATA_LANE 16 // 16 byte -> 128 bit di una linea

void print_register(__m128i reg)
{
    // Array temporaneo per memorizzare il contenuto del registro
    int8_t temp[VECTOR_LENGTH] __attribute__((aligned(SSE_DATA_LANE)));

    // Memorizza il contenuto del registro nell'array temporaneo
    _mm_store_si128((__m128i*)temp, reg);

    // Stampa il contenuto dell'array
    printf("Contenuto del registro: ");
    for (int i = 0; i < VECTOR_LENGTH; i++)
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
{ /*
    // Due array di int8_t (8 bit con segno), allineati su 16 byte
    int8_t A[VECTOR_LENGTH] __attribute__((aligned(SSE_DATA_LANE))); 
    int8_t MAX[VECTOR_LENGTH] __attribute__((aligned(SSE_DATA_LANE)));
    int8_t MIN[VECTOR_LENGTH] __attribute__((aligned(SSE_DATA_LANE)));
    // Puntatori ai registri per operazioni SIMD
    __m128i XMM_SSE_REG;
    __m128i XMM_SSE_REG_MAX;
    __m128i XMM_SSE_REG_MIN;
    __m128i *RegArrey = (__m128i*) A;
    __m128i *RegMax = (__m128i*) MAX;
    __m128i *RegMin = (__m128i*) MIN;
    // Inizializzazione degli array
    for (int i = 0; i < VECTOR_LENGTH; i++)
    {
        A[i] = i;
        MAX[i] = 127;
        MIN[i] = -128;
    }
    A[8] = 127;

    printf("\nInput data:\n");
    print_output(A, VECTOR_LENGTH);

    printf("\nMax:\n");
    print_output(MAX, VECTOR_LENGTH);

    printf("\nMin:\n");
    print_output(MIN, VECTOR_LENGTH);

    // Caricamento dei dati nei registri SIMD
    XMM_SSE_REG = _mm_load_si128(RegArrey);
    XMM_SSE_REG_MAX = _mm_load_si128(RegMax);
    XMM_SSE_REG_MIN = _mm_load_si128(RegMin);
    
    //print_register(XMM_SSE_REG_MAX);
    //print_register(XMM_SSE_REG_MIN);
    print_register(XMM_SSE_REG);

    __m128i time1 = _mm_cmpgt_epi8(XMM_SSE_REG_MAX, XMM_SSE_REG);
    print_register_bit(time1);

   __m128i time2 = _mm_blendv_epi8(XMM_SSE_REG_MAX, XMM_SSE_REG, time1);
    print_register(time2);
    __m128i XOR = _mm_xor_si128 (XMM_SSE_REG_MAX, time2);
     print_register(XOR);
    // Il resto del codice per le operazioni SIMD può seguire qui...
    //while(true){

    //}
    */
    //prev_mins
    //new_data
    //voglio estrarre solo i valori minimi fra i due registri per ogni linea e metterni in un nuovo registro 
    /*
    int8_t NEW_DATA[VECTOR_LENGTH] __attribute__((aligned(SSE_DATA_LANE)));
    int8_t PREV_MINS[VECTOR_LENGTH] __attribute__((aligned(SSE_DATA_LANE)));
    //registri SIMD
    __m128i new_data;
    __m128i prev_mins;
    //inizializziamo
    for (int i = 0; i < VECTOR_LENGTH; i++) {
        NEW_DATA[i] = i;         // Esempio: riempi NEW_DATA con valori da 0 a 15
        PREV_MINS[i] = -128 + i; // Esempio: riempi PREV_MINS con valori da -128 a -113
    }
    PREV_MINS[0]=127;
    NEW_DATA[0]=1;
    // Puntatori agli array per le operazioni SIMD
    __m128i *new_data_puntatore = (__m128i*) NEW_DATA;
    __m128i *prev_mins_puntatore = (__m128i*) PREV_MINS;
    //facciamo la load degli arrey sui registri
    new_data = _mm_load_si128(new_data_puntatore);
    prev_mins = _mm_load_si128(prev_mins_puntatore);
    print_register(new_data);
    print_register(prev_mins);
    //La domanda che ci facciamo è new_data < prev_mins -> 1(si)  0(no)
    __m128i Mask1 = _mm_cmplt_epi8(new_data, prev_mins);
     print_register_bit(Mask1);

     //usando operazione logica AND eliminiamo i dati che non ci interessano da new_data [non sono minori di quelli in prev_mins]
     //Mask1 -> maschera per AND
    __m128i temp1 = _mm_and_si128(Mask1,new_data); //filtriamo i vari di new_data < di  prev_mins
    print_register(temp1);
    //i valori all'interno di temp1 != 0 sono quelli che sostituiranno quelli più grandi in prev_mins

    //analogamente a come ho filtrato i valori da new_data devo fare lo stesso per prev_mins
    //usiamo NOTAND
    __m128i temp2 = _mm_andnot_si128(Mask1, prev_mins); //filtriamo i vari di prev_mins < di  new_data
    print_register(temp2);

    //aggiorniamo il valore di prev_mins fondendo i due temp_1 temp_2 che rappresentano i valori minori filtrati linea per linea dei due registri
    //salviamo tutto dentro prev_mins
    prev_mins = _mm_or_si128(temp1, temp2); //filtriamo i vari di prev_mins < di  new_data
    print_register(prev_mins);
    return 0;*/
   // __m128i all_ones = _mm_set1_epi8(-1);
    __m128i uno = _mm_set1_epi8(1);
    int8_t DATA[VECTOR_LENGTH] __attribute__((aligned(SSE_DATA_LANE)));
    int8_t MAX[VECTOR_LENGTH] __attribute__((aligned(SSE_DATA_LANE)));
    //registri SIMD
    __m128i data;
    __m128i max;
    //inizializziamo
    for (int i = 0; i < VECTOR_LENGTH; i++) {
        DATA[i] = i;         // Esempio: riempi NEW_DATA con valori da 0 a 15
        MAX[i] = 127; // Esempio: riempi PREV_MINS con valori da -128 a -113
    }
    DATA[0]=124;
    DATA[2]=120;    
    // Puntatori agli array per le operazioni SIMD
    data = _mm_load_si128((__m128i*)DATA);
    max = _mm_load_si128((__m128i*)MAX);

   // print_register(data);
   // print_register(max);

// Creazione della maschera con _mm_cmplt_epi8 (data < max)
//__m128i mask = _mm_cmplt_epi8(data, max);
//print_register_bit(mask);

// Esegui l'AND tra data e mask
//data = _mm_and_si128(data, mask);
//printf("\nDopo AND\n");
//print_register(data);

// Utilizza _mm_blendv_epi8 con i parametri scambiati per evitare di invertire la maschera
//__m128i newMax = _mm_blendv_epi8(data, max, mask);
//printf("\nDopo Blendv\n");
//print_register(newMax);
//if (!(_mm_test_all_ones(_mm_blendv_epi8(data, max, mask)))){
//printf("\nTrovato il MAX\n");
////}

int i=0;
/*
do{
mask = _mm_cmplt_epi8(data, max);
printf("\n mask all'iterazione %d\n", i);
print_register_bit(mask);
data = _mm_and_si128(data, mask);
printf("\n Data dopo AND all'iterazione %d\n", i);
print_register_bit(data);
max = _mm_blendv_epi8(data, max, mask);
printf("\n nuovo max\n");
print_register(max);
}while(_mm_test_all_ones(max));
*/
    __m128i mask;
    u_int64_t clock_counter_start = __rdtsc();
    do {
        //data<max 
        mask = _mm_cmplt_epi8(data, max); // Crea una maschera per verificare se data è minore di max
        //print_register_bit(mask);
        if (!_mm_test_all_ones(mask)) { // Esci dal ciclo se non ci sono più elementi minori
            break; // Se non ci sono più elementi minori, esci dal ciclo
        }
        // Altrimenti, continua a sottrarre da max
        max = _mm_sub_epi8(max, uno);  // Sottrae 1 da ogni elemento del registro max
    } while (1); // Continua indefinitamente finché non esce
    u_int64_t clock_counter_end = __rdtsc();
    //printf("\nElapsed clocks1: %lu\n", clock_counter_end-clock_counter_start);
    //printf("\nMassimo trovato:\n");
    //print_register(max);
    //Versione sequenziale 
    int maxint=0;
    int NEW_DATA[VECTOR_LENGTH];
    for (int i=0; i<VECTOR_LENGTH; i++){
        NEW_DATA[i]=i;
    }
    clock_counter_start = __rdtsc();
    for (int i = 0; i < VECTOR_LENGTH; i++) {
        if (NEW_DATA[i] > maxint) {
            maxint = NEW_DATA[i]; // Aggiorna il massimo se il valore corrente è maggiore
        }
    }
     clock_counter_end = __rdtsc();
     /*
    //printf("\nElapsed clocks2: %lu\n", clock_counter_end-clock_counter_start);
    data = _mm_load_si128((__m128i*)DATA);
    max = _mm_load_si128((__m128i*)MAX);
    printf("\ntroviamo il massimo del registro\n");
    print_register(data);
    mask = _mm_cmplt_epi8(data, max);
    print_register_bit(mask);

    __m128i prova=_mm_xor_si128(data,max);
   // print_register(prova);
    printf("\n XOR data/max\n");
    print_register_bit(prova);

    prova=_mm_and_si128(data,max);
   // print_register(prova);
    printf("\n and data/max\n");
    print_register_bit(prova);

     prova=_mm_or_si128(data,max);
   // print_register(prova);
    printf("\n OR data/max\n");
    print_register_bit(prova);

     prova=_mm_andnot_si128(data,mask);
   // print_register(prova);
    printf("\n ANDNOT data/max\n");
    print_register_bit(prova);*/
     for (int i = 0; i < VECTOR_LENGTH; i++) {
        if(i<VECTOR_LENGTH/2){
            MAX[i] = 127;
        }else{
            MAX[i] = 0;
        }
        DATA[i] = i;         // Esempio: riempi NEW_DATA con valori da 0 a 15
    }
     DATA[0]=124;
    DATA[2]=120;    
    data = _mm_load_si128((__m128i*)DATA);
    max = _mm_load_si128((__m128i*)MAX);
    //print_register(data);
    //print_register(max);
    /*
     clock_counter_start = __rdtsc();
    mask = _mm_cmplt_epi8(data, max);
    //print_register_bit(mask);
     __m128i meta2=_mm_shuffle_epi32(_mm_and_si128(data, mask), _MM_SHUFFLE(0, 1, 2, 3));
    // print_register(meta2);
     __m128i meta1=_mm_andnot_si128(mask,data );
     //print_register(meta1);
     data=_mm_max_epi8(meta1,meta2 );
    // print_register(data);
     mask=_mm_shuffle_epi32(mask, _MM_SHUFFLE(0, 2, 1, 3));
    // print_register_bit(mask);
     meta2=_mm_andnot_si128(mask,data );
   // print_register(_mm_shuffle_epi32(meta2,_MM_SHUFFLE(3, 0, 2, 1)));
    meta1=_mm_shuffle_epi32(_mm_and_si128(data, mask), _MM_SHUFFLE(3, 1, 2, 0));
   // print_register(meta1);
    data=_mm_max_epi8(meta1,_mm_shuffle_epi32(meta2,_MM_SHUFFLE(3, 0, 2, 1)) );
    //print_register(data);
     clock_counter_end = __rdtsc();
     printf("\nElapsed clocks2: %lu\n", clock_counter_end-clock_counter_start);*/
     clock_counter_start = __rdtsc();
     //print_register(data);
     //print_register_bit(max);
     __m128i data1=_mm_and_si128(data,max );
     __m128i data2=_mm_and_si128(_mm_shuffle_epi32(data,_MM_SHUFFLE(0, 1, 2, 3)), max);
    // print_register(data1);
    // print_register(data2);
     data =_mm_max_epi8(data1,data2 );
     //print_register(data);
     clock_counter_end = __rdtsc();
     printf("\nElapsed clocks2: %lu\n", clock_counter_end-clock_counter_start);
    return 0;
}

