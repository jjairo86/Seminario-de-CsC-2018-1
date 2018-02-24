import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
from random import random

N = 1000000
BLOCKS = 1
THREADS = N

mod = SourceModule("""
    #include <stdio.h>
    #define N 32
    __global__ void montecarlo(float *x, float *y,float *contador) {
        int indice = threadIdx.x + blockIdx.x*blockDim.x;
        if((x[indice]*x[indice] + y[indice]*y[indice]) <=1.0) {
            atomicAdd(contador,1);//contador++;
        }
    } 
""")

x = np.array([random() for i in range(N)]).astype(np.float32) 
y = np.array([random() for i in range(N)]).astype(np.float32)
contador = np.array([0]).astype(np.float32)

montecarlo = mod.get_function("montecarlo")

montecarlo(cuda.In(x), cuda.In(y), cuda.Out(contador), 
    block=(BLOCKS,1,1), grid=(THREADS,1,1) )

#print (x)
#print (y)
#print (contador)
pi = (float(contador) / N) * 4
print ("PI = ",pi)