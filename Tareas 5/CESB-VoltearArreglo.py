import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np


mod = SourceModule("""
    #include <stdio.h>
    __global__ void voltearArreglo(float *a, int *dimension){
        float auxiliar=0;
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if(index<=dimension[0]/2){
            int index2 = dimension[0]-1 - index;
            auxiliar=a[index];
            a[index]=a[index2];
            a[index2]=auxiliar;
        }
    }
""")
N = 1000
THREADS = 10

a = np.array([i+1 for i in range(N)]).astype(np.float32) 
dimension = np.array(N).astype(np.int32)
funcion = mod.get_function("voltearArreglo")


print(a)

funcion(cuda.InOut(a), cuda.In(dimension), 
    block=(THREADS,1,1), grid=(N//THREADS,1,1) )

print(a)