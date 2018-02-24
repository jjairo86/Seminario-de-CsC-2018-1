"""import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

N = 32
BLOCKS = 2
THREADS = 16

mod = SourceModule(
    #include <stdio.h>
    #define N 32
    __global__ void montecarlo(float *a, float *b,float *c) {
        int indice = threadIdx.x + blockIdx.x*blockDim.x;
        if((x[indice]*x[indice] + y[indice]*y[indice]) <=1.0) {
            atomicAdd(contador,1);//contador++;
            //printf("Contador: %d\n",*contador);
        }
    } 
"""#)
import numpy as np
#x = np.array([1 for i in range(N*N)]).astype(np.float32) 
#y = np.array([i for i in range(N*N)]).astype(np.float32)
x = np.random.randn(1000).astype(np.float32)
y = np.random.randn(1000).astype(np.float32)
contador = np.array([0]).astype(np.float32)
"""
montecarlo = mod.get_function("montecarlo")

montecarlo(cuda.In(a), cuda.In(b), cuda.Out(c), 
    block=(BLOCKS,BLOCKS,1), grid=(THREADS,THREADS,1) )
#print( np.sum(a*b) )
print( c )"""
print (x)
print (y)
print (contador)