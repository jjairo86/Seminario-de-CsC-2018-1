import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

M = 1024
N = 1024
THREADS_PER_BLOCK = 32

mod = SourceModule("""
    #define N 1024
    #define M 1024

    __global__ void transpuestaFunc(float *a, float *b) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;
        int x = j+i*M;
        int y = i+j*N;

        if(x < N*M){
            b[y] = a[x];
        } 
    }
""")

a = np.array([i for i in range(N*M)]).astype(np.float32) 
b = np.array([0 for i in range(M*N)]).astype(np.float32)
c = np.array([0 for i in range(N*M)]).astype(np.float32)

transpuesta = mod.get_function("transpuestaFunc")

transpuesta(cuda.In(a), cuda.Out(b), block=(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1), grid=(M//THREADS_PER_BLOCK, N//THREADS_PER_BLOCK, 1))

print( b.astype(np.int32).reshape((M,N)) )