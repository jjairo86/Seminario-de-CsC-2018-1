import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

N = 32
BLOCKS = 2
THREADS = 16

mod = SourceModule("""
    #include <stdio.h>
    #define N 32
    __global__ void transpuesta(float *a, float *b) {
        int i = threadIdx.x + blockIdx.x*blockDim.x; 
        int j = threadIdx.y + blockIdx.y*blockDim.y; 
        //printf("blockIdx.x: %d blockDim.x: %d threadIdx: %d\\n",blockIdx.x,blockDim.x,threadIdx.x);
     
        //if(i==j)printf("[%d][%d]\\n",i,j);
        
        while(i<N){
            j = threadIdx.y + blockIdx.y*blockDim.y;
            while(j<N){
                b[j*N+i]=a[i*N+j];
                j+= blockDim.y*gridDim.y;
            }
            i+=blockDim.x*gridDim.x;
        } 
    } 
""")

a = np.array([i for i in range(N*N)]).astype(np.float32) 
b = np.array([0 for i in range(N*N)]).astype(np.float32)

transpuesta = mod.get_function("transpuesta")

transpuesta(cuda.In(a), cuda.Out(b), 
    block=(BLOCKS,BLOCKS,1), grid=(THREADS,THREADS,1) )
print( a )
print( b )