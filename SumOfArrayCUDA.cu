//!nvcc -o SumOfArray SumOfArray.cu
//!./SumOfArray
#include <stdio.h>
__global__ void hello_cuda(float *A, float *B, float *C) { 
  //per individuare univocamente la posizione della cella nello spazioni usiamo VETTORI
  //INDICE GLOBALE 
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < 1000) {  // Aggiunto controllo per evitare accessi fuori dai limiti
    printf("Indirizzo Globale -> %d, Blocco -> %d, thread del blocco -> %d \n", idx,blockIdx.x,threadIdx.x );
    C[idx] = A[idx] + B[idx];
  }
  }
int main() {
//CPU
  //qui abbiamo fatto un esempio con un 5 thread di un solo blocco
  int N=100;
  int size = N *sizeof(float);
  float *A, *B, *C;
  A=(float*)malloc(size);
  B=(float*)malloc(size);
  C=(float*)malloc(size);
  for (int i = 0; i < N; i++) {
    A[i] = i;  // Ad esempio 0.0, 1.0, 2.0, 3.0, 4.0
    B[i] = i;  // Ad esempio 0.0, 2.0, 4.0, 6.0, 8.0
  }
//GPU
//il kernel della GPU non vede i dati allocati per la CPU allora bisogna
//allocare nella CPU, inizializzare nella GPU e poi spostare tutto nella GPU
  float *d_A, *d_B, *d_C; 
  // Allochiamo la memoria sul device
  cudaMalloc((void**)&d_A, size);
  cudaMalloc((void**)&d_B, size);
  cudaMalloc((void**)&d_C, size);
// copia dati host ->device
  cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
  //voglio sapere in quanti blocchi distribuire gli N thread 
  int blockSize = 10;
  int gridSize = (N + blockSize - 1) / blockSize;
  hello_cuda<<<gridSize, blockSize>>>(d_A,d_B,d_C); 
  cudaDeviceSynchronize();
  //copiamo C da device -> host 
  cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
   printf("Risultato della somma:\n");
  for (int i = 0; i < N; i++) {
    printf("C[%d] = %f\n", i, C[i]);
  }
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(A);
  free(B);
  free(C);
return 0;
}
