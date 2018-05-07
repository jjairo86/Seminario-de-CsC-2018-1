import numpy as np
from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit
from pylab import plot, savefig, grid, xlabel, ylabel

largo = 128


kernel = """
#include<math.h>

	__global__ void espejo(float *x){
	    int i = threadIdx.x;
	    int dim = %(dim)s;

	    x[i] = x[absoluto(i - dim + 1)];
}"""


kernel = kernel % {
    'dim'  :   largo,
    'ABS'   :   absoluto
}

arreglo = np.arange(size)
in_gpu = gpuarray.to_gpu(arreglo.astype(np.float32))
mod = compiler.SourceModule(kernel)
func = mod.get_function('espejo')

func(in_gpu, block = (size,1,1),grid = (1,1,1))

imprimir = in_gpu.get()
print( imprimir )
