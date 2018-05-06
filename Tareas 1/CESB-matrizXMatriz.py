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
    __global__ void multMatrices(float *a, float *b,float *c) {
        int i = threadIdx.x + blockIdx.x*blockDim.x; 
        int j = threadIdx.y + blockIdx.y*blockDim.y; 
        
        c[j+i*N] = 0; // 4,194,303

        for(int k=0 ; k < N ; k++ ){
            c[j+i*N] += a[k+i*N] * b[j+k*N];
        }
    } 
""")

a = np.array([1 for i in range(N*N)]).astype(np.float32) 
b = np.array([1 for i in range(N*N)]).astype(np.float32)
c = np.array([0 for i in range(N*N)]).astype(np.float32)

multMatrices = mod.get_function("multMatrices")

multMatrices(cuda.In(a), cuda.In(b), cuda.Out(c), 
    block=(BLOCKS,BLOCKS,1), grid=(THREADS,THREADS,1) )
#print( np.sum(a*b) )
print( c )