import numpy as np
from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit
from pylab import plot, savefig, grid, xlabel, ylabel

M = 256


kernel = """
#include<math.h>
__global__ void criba(float *global){
    int tidx = threadIdx.x;
    int x = %(dim)s;
    if( tidx < x/2 && tidx > 1 ){
        for(int i = 2; tidx * i < x; i++){
            global[tidx * i] = 0;
        }
    }
}"""



kernel = kernel % {
    'dim'  :   M,
    'aux'    :   0
}

n_sqrt = np.floor(np.sqrt(n))
array = np.arange(n)
array[0] = 1
in_gpu = gpuarray.to_gpu(array.astype(np.float32))

mod = compiler.SourceModule(kernel)

criba = mod.get_function('criba')
criba(in_gpu, block = (n,1,1),grid = (1,1,1))
array = []
for i in in_gpu.get():
    if i != 0:
        array.append(i)
print(array)
