#include <stdio.h>
#define N 128*128


__global__ void kernelMontecarlo(float *x, float *y,int *contador) {
    //int i = threadIdx.x + blockIdx.x*blockDim.x; 
    //int j = threadIdx.y + blockIdx.y*blockDim.y; 
    int indice = threadIdx.x + blockIdx.x*blockDim.x;
	//int indice=i;
	//printf("Indice: %f\n",(x[indice]*x[indice] + y[indice]*y[indice]));
	if((x[indice]*x[indice] + y[indice]*y[indice]) <=1.0) {
		atomicAdd(contador,1);//contador++;
		//printf("Contador: %d\n",*contador);
	}
} 

float calcularPI();
int main(){
    //inicia semilla
    float resultado=calcularPI();
	printf("\nPi: %f\n",resultado);
    return 0;
}
float calcularPI(){
    float x[N],y[N];
    int contador;
    float *xD,*yD;
    int *contadorD=0;
    int size = sizeof(float)*N; 
	//crear Eventos
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
	/*dim3 nb(B,B); 
    dim3 nh(N/T,N/T);
    */
	srand(time(NULL));
	//memoria
	memset(x, 0, N*sizeof(float));
	memset(y, 0, N*sizeof(float));
	//genera numeros
	for(int i=0;i<N;i++){
		x[i]=rand()/(RAND_MAX + 1.0f);	
		y[i]=rand()/(RAND_MAX + 1.0f);
		//x[i]=(float)(rand()%100) / 99;	
		//y[i]=(float)(rand()%100) / 99;
	}
	cudaMalloc(&xD, size); 
	cudaMalloc(&yD, size);
	cudaMalloc(&contadorD, sizeof(int));
	contador=0;
	cudaMemcpy(xD, x, size, cudaMemcpyHostToDevice); 
    cudaMemcpy(yD, y, size, cudaMemcpyHostToDevice);  
	cudaMemcpy(contadorD, &contador, sizeof(int), cudaMemcpyHostToDevice); 
	//inicio
    cudaEventRecord(start);
	kernelMontecarlo<<< N/32, 32>>>(xD, yD,contadorD); 
	cudaMemcpy(&contador, contadorD, sizeof(int), cudaMemcpyDeviceToHost); 
	//for(int i=0; i<N;i++) 
	//	if((x[i]*x[i] + y[i]*y[i]) <=1.0) contador++;
	//float resultado= 4*contador/N;	
	//final
	cudaEventRecord(stop);
	
	//sincronizar
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Tiempo: %f milisegundos\n",milliseconds);
	float resultado=4*contador/float(N);
	return resultado;
}