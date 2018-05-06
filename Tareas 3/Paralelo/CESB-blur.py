import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt

THREADS = 32

mod = SourceModule("""
    #include <stdio.h>
    #define R 1
    __global__ void blur(float *a,float *matriz, int *dimensiones) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;
        if (i < dimensiones[0] && j < dimensiones[1]) {
            //--------------CAlCULA LOS INDICES
            //el indice del centro
            int index4 = (j + i*dimensiones[1])*3;
            //el indice derecha
            int index5 = (j+3 + i*dimensiones[1])*3;
            //el indice derecha
            int index3 = (j-3 + i*dimensiones[1])*3;
            //el indice de arriba
            int index1 = ((j) + (i-1)*dimensiones[1])*3;
            //el indice de aabajo
            int index7 = ((j) + (i+1)*dimensiones[1])*3;
            //el indice dela esquina superior izquirda
            int index0 = ((j-3) + (i-1)*dimensiones[1])*3;
            //el indice dela esquina superior derecha
            int index2 = ((j+3) + (i-1)*dimensiones[1])*3;
            //el indice dela esquina inferior izquirda
            int index6 = ((j-3) + (i+1)*dimensiones[1])*3;
            //el indice dela esquina superior derecha
            int index8 = ((j+3) + (i+1)*dimensiones[1])*3;
            //------------------------------------------   
            a[index4] = matriz[0]*a[index0]+matriz[1]*a[index1]+matriz[2]*a[index2]+matriz[3]*a[index3]+matriz[4]*a[index4]+matriz[5]*a[index5]+matriz[6]*a[index6]+matriz[7]*a[index7]+matriz[8]*a[index8];
            a[index4+1] = matriz[0]*a[index0+1]+matriz[1]*a[index1+1]+matriz[2]*a[index2+1]+matriz[3]*a[index3+1]+matriz[4]*a[index4+1]+matriz[5]*a[index5+1]+matriz[6]*a[index6+1]+matriz[7]*a[index7+1]+matriz[8]*a[index8+1];
            a[index4+2] = matriz[0]*a[index0+2]+matriz[1]*a[index1+2]+matriz[2]*a[index2+2]+matriz[3]*a[index3+2]+matriz[4]*a[index4+2]+matriz[5]*a[index5+2]+matriz[6]*a[index6+2]+matriz[7]*a[index7+2]+matriz[8]*a[index8+2];    
        }
    } 
""")

aBlur = mod.get_function("blur")

#aqui leo la imagen
imagen= misc.imread("a.jpeg").astype(np.float32)
#obtengo los valores de laas dimensiones de la matriz y el tres es los 3 RGB XD
x,y,tres=imagen.shape
dimensiones = np.array([x, y]).astype(np.int32)
#matriz de tranformar Xd
matriz=np.array([1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9]).astype(np.float32)
#matriz=np.array([0,-1,0,-1,5,-1,0,-1,0]).astype(np.float32)
#lo pongo en una dimension
imagen=imagen.reshape(x*y*tres)
#aqui usa la funcion para cambiar a grises luminosity
#imagenGray=toGrayLuminosity(imagen,x*y*tres)
aBlur(cuda.InOut(imagen),cuda.In(matriz),cuda.In(dimensiones), 
    block=(THREADS,THREADS,1), grid=(int((dimensiones[0]+THREADS)//THREADS),int((dimensiones[1]+THREADS)//THREADS),1 ))
#cambia a las 3 dimensines anteriores para mostrar la nueva grises
imagen=imagen.reshape(x,y,tres)
misc.imsave('borrosa.png', imagen)

