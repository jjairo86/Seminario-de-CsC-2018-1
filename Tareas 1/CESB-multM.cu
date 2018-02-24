#include <stdio.h>
#include <cuda.h>

#define M 512
#define N 512 
#define T 256
#define B 4
float a[N][N], b[N][N], c[N][N]; 

__global__ void kernelMultMatrices(float *a, float *b, float *c,int m, int n) {
    int i = threadIdx.x + blockIdx.x*blockDim.x; 
    int j = threadIdx.y + blockIdx.y*blockDim.y; 
    //printf("%d,%d\n",i,j); 
    c[j+i*n]=0;
    for(int k=0;k<N;k++) c[j+i*n]+=a[j+k*n]*b[k+i*n];;
    __syncthreads();
} 


void multMatrices(float *a, float *b, float *c, int m, int n) {
    float *ad, *bd, *cd; 
    int size = sizeof(float)*m*n; 
    dim3 nb(B,B); 
    //dim3 nh(N/T,N/T);
    dim3 nh(2048/512,2048/512);
    //printf("%d, %d\n",nb.x,nb.y); 
    //printf("%d, %d\n",nh.x,nh.y); 
    //crear Eventos
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
     

    cudaMalloc(&ad, size); 
    cudaMalloc(&bd, size); 
    cudaMalloc(&cd, size); 

    cudaMemcpy(ad, a, size, cudaMemcpyHostToDevice); 
    cudaMemcpy(bd, b, size, cudaMemcpyHostToDevice); 

    //inicio
    cudaEventRecord(start);

    kernelMultMatrices<<<nb , nh>>>(ad, bd,cd, m, n); 
    //final
    cudaEventRecord(stop);

    cudaMemcpy(c, cd, size, cudaMemcpyDeviceToHost); 

    //sincronizar
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Tiempo: %f milisegundos\n",milliseconds);
    
    cudaFree(ad); cudaFree(bd); cudaFree(cd); 
}

int main() {
    int i, j; 

    
    for (i=0; i<M; i++) {
       for (j=0; j<N; j++) {
          a[i][j]=b[i][j]=1;
          c[i][j]=1;
       }
    }
    multMatrices((float *)a, (float *)b, (float *)c,N, N); 
    printf("A\n"); 
    for (i=0; i<10; i++) {
        for (j=0; j<10; j++) {
          printf("%4.2f  ", a[i][j]);
        }
        printf("\n"); 
     }
     printf("B\n");
     for (i=0; i<10; i++) {
        for (j=0; j<10; j++) {
          printf("%4.2f  ", b[i][j]);
        }
        printf("\n"); 
     }
     printf("C\n");
    for (i=0; i<10; i++) {
       for (j=0; j<10; j++) {
         printf("%4.2f  ", c[i][j]);
       }
       printf("\n"); 
    }
  
}


