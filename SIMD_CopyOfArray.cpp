#include <stdio.h>
#include <immintrin.h> //intrinseci di Intel x86
#define VECTOR_LENGTH 32 //array statici di 32 elementi 
//Ampiezza dei registri 
#define SSE_DATA_LANE 16 //16byte -> 128bit di una linea 
#define DATA_SIZE 1 //size del dato 

void print_output(char *A, char *B, int length)
{
 for (int i=0; i<VECTOR_LENGTH; i++)
 {
 printf("A[%d]=%d, B[%d]=%d\n",i,A[i],i,B[i]);
 }
}
int main()
{
//due array char[byte con segno], li creo allineati sui 16byte 
 char A[VECTOR_LENGTH] __attribute__((aligned(SSE_DATA_LANE))); // 16 byte (128 bit) aligned
 char B[VECTOR_LENGTH] __attribute__((aligned(SSE_DATA_LANE))); // 16 byte (128 bit) aligned
 //due puntatori ai due array di tipo registro esteso richiesto dalla load
 //rispettivamente agli indirizzi di partenza di A e B
 __m128i *p_A = (__m128i*) A;
 __m128i *p_B = (__m128i*) B;
 
 __m128i XMM_SSE_REG;
for (int i=0; i<VECTOR_LENGTH; i++)
{
 A[i]=i;
 B[i]=0;
}
printf("\nInput data:\n");
print_output(A,B,VECTOR_LENGTH);
//Attenzio scandiamo array di 32 elementi 
//devo dare 2 loead e 2 store si 16byte ciascuno 
for (int i=0; i<VECTOR_LENGTH/SSE_DATA_LANE/DATA_SIZE; i++) //ho due sole iterazioni di 16 elementi ciascuno 
{
//la load è allineata 
 XMM_SSE_REG = _mm_load_si128 (p_A+i);
 //faccio la store su B del dato appena letto in A e lo store in B
 _mm_store_si128 (p_B+i, XMM_SSE_REG);
}
printf("\nOutput data:\n");
print_output(A,B,VECTOR_LENGTH);
}
