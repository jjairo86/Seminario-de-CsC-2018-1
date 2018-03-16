import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt

THREADS = 32

mod = SourceModule("""
    #include <stdio.h>
    __global__ void cambiarAGrises(float *a, int *dimensiones) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;
        if (i < dimensiones[0] && j < dimensiones[1]) {
            int index = j + i*dimensiones[1]; 
            int index3 = index*3;   
            float factor = 0.21*a[index3] + 0.72*a[index3+1] + 0.07*a[index3+2];
            a[index3] = factor;
            a[index3+1] = factor;
            a[index3+2] = factor;
        }
    } 
""")

#pa guardar
#misc.imsave('name.png', name) 
#Funcion escala de grises Luminosity
def toGrayLuminosity(matriz, dim):
    i=0
    nueva=matriz.copy()
    while i<dim-3:
        #R * 0.2989 +  G * 0.5870 + B * 0.1140 cambiar por esos valores
        factor=0.21*nueva[i] + 0.72*nueva[i+1] + 0.07*nueva[i+2]
        nueva[i]=factor
        nueva[i+1]=factor
        nueva[i+2]=factor
        i+=3
    return nueva  

aGris = mod.get_function("cambiarAGrises")

#aqui leo la imagen
imagen= misc.imread("a.jpeg").astype(np.float32)
#obtengo los valores de laas dimensiones de la matriz y el tres es los 3 RGB XD
x,y,tres=imagen.shape
dimensiones = np.array([x, y]).astype(np.int32)
#lo pongo en una dimension
imagen=imagen.reshape(x*y*tres)
#aqui usa la funcion para cambiar a grises luminosity
#imagenGray=toGrayLuminosity(imagen,x*y*tres)
aGris(cuda.InOut(imagen),cuda.In(dimensiones), 
    block=(THREADS,THREADS,1), grid=(int((dimensiones[0]+THREADS)//THREADS),int((dimensiones[1]+THREADS)//THREADS),1 ))
#cambia a las 3 dimensines anteriores para mostrar la nueva grises
imagen=imagen.reshape(x,y,tres)
#cambia a las 3 dimensines anteriores para mostrar la original
#imagen=imagen.reshape(x,y,tres)
#ver antes
#plt.imshow(imagen)
#plt.show()
#ver despues grises
#plt.imshow(imagenGray)
#plt.show()
#misc.imsave('original.png', imagen)
misc.imsave('gris.png', imagen)

