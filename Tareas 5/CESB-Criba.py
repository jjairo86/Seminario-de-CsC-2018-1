import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np


mod = SourceModule("""
    #include <stdio.h>
    __global__ void criba(float *a, float *primos, int *dimension){
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        int numero=a[index];
        if(numero+index<=dimension[0]-1){
            for(int j=index+numero;j<=dimension[0];j+=numero){
                if(primos[j]!=0){
                    primos[j]=0;
                }
            }
        }
    }
""")
N = 1000
THREADS = N-1

a = np.array([i for i in range(2,N+1)]).astype(np.float32)
marcas = np.array([1 for i in range(2,N+1)]).astype(np.float32)  
dimension = np.array(N-1).astype(np.int32)
funcion = mod.get_function("criba")


funcion(cuda.In(a), cuda.InOut(marcas), cuda.In(dimension), 
    block=(THREADS,1,1), grid=((N-1)//THREADS,1,1) )
primos=np.array([a[i] for i in range(N-1) if marcas[i]==1]).astype(np.int32)
print(f"Los números primos hasta {N} son :\n\n",primos)