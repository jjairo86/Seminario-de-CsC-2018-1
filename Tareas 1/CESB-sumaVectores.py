import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np


mod = SourceModule("""
__global__ void kernelSumaVectores(float *destino, float *a, float *b)
{
  int i = threadIdx.x;
  destino[i] = a[i] * b[i];
}
""")

kernelSumaVectores = mod.get_function("kernelSumaVectores")

a = np.random.randn(1000).astype(np.float32)
b = np.random.randn(1000).astype(np.float32)
#a = np.array([1,2,3,4,5]).astype(np.float32)
#b = np.array([5,4,3,2,1]).astype(np.float32)


destino = np.zeros_like(a)
kernelSumaVectores(
        cuda.Out(destino), cuda.In(a), cuda.In(b),
        block=(1000,1,1), grid=(1,1))

#print (destino-a*b)
print (destino)
"""

"""