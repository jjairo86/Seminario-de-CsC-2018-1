import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

N = 32
threads_per_block = 8

mod = SourceModule("""
    const int L = 32;
    __global__ void multCRS(float *val, int *col_ind, int *row_ptr, float *u, float *resultado){
        int i = threadIdx.x + blockIdx.x*blockDim.x; 
        int j = threadIdx.y + blockIdx.y*blockDim.y;
        float suma = 0.0;
        for(int k = row_ptr[i]-1; k < row_ptr[i+1]-1; k++){
            suma += val[k] * u[j + ( (col_ind[k]-1) * L) ];
        }
        resultado[j+i*L] = suma;
    }
""")

matriz1 = [ (1 if j == i else 0) for i in range(N) for j in range(N) ]
matriz2 = np.array([1 for i in range(N*N)]).astype(np.float32)
matriz3 = np.array([0 for i in range(N*N)]).astype(np.float32)
nnz = sum([ (1 if matriz1[j+i*N] != 0 else 0) for i in range(N) for j in range(N)]) 
##############3
# inicializo val, col_ind y row_ptr
val = np.array([1 for i in range(nnz)]).astype(np.float32)
col_ind = np.array([1 for i in range(nnz)]).astype(np.float32)
row_ptr = np.array([1 for i in range(N+1)]).astype(np.float32)
# obtener val y col_ind
contador = 0
for i in range(N):
    for j in range(N):
        if matriz1[j+i*N] != 0:
            val[contador], col_ind[contador] = matriz1[j+i*N], j+1
            contador += 1
# obtengo row_ptr
contador, indice, fila = 0, 0, -1
for i in range(N):
    for j in range(N):
        if matriz1[j+i*N] != 0:
            contador+=1
            if i != fila:
                row_ptr[indice], fila = contador, i
                indice += 1
row_ptr[N] = nnz+1
#llena los datos para crs
val=np.array(val).astype(np.float32)
col_ind=np.array(col_ind).astype(np.int32)
row_ptr =np.array(row_ptr).astype(np.int32)
##############
# get kernel
multiplicarCRS= mod.get_function("multCRS")
# mult
multiplicarCRS(cuda.In(val), cuda.In(col_ind), cuda.In(row_ptr), cuda.In(matriz2), cuda.Out(matriz3), 
    block=(threads_per_block, threads_per_block, 1), grid=(N//threads_per_block, N//threads_per_block, 1))
print(matriz1)
print(matriz2)
print(matriz3)