import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt

THREADS = 32


"""
*Tiene fallas, con imagenes grandes no ha funcionado
"""

mod = SourceModule("""
    #include <stdio.h>
    #include <math.h>
    __global__ void upScaling(float *a,float *matriz, int *dimensiones) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;
    
        if (i < dimensiones[2] && j < dimensiones[3]) {
            int factor=dimensiones[4];
            int salto=dimensiones[1]*factor*3;
            int x=int(i/factor);
            int y=int(j/factor);
            //para la nueva imagen
            int indexNuevo=(y+x*dimensiones[3])*factor*3;
            //para moverse en la imagen original
            int indexOriginal=(y+x*dimensiones[1])*3;
            //para los renglones
            for(int r=0;r<=factor;r++){
                //para las colunmas
                for(int s=0;s<=factor;s++){
                    //para el rgb
                    for(int k=0;k<=2;k++){
                        matriz[(3*s)+(indexNuevo+salto*r)+k]=a[indexOriginal+k];
                    }
                }
            }
        }
    } 
""")

funcion = mod.get_function("upScaling")
#el factor por el cual se multiplican las dimensiones
factor=2
#aqui leo la imagen
imagen= misc.imread("cuadros.jpg").astype(np.float32)
#obtengo los valores de laas dimensiones de la matriz y el tres es los 3 RGB XD
x,y,tres=imagen.shape
nuevaDimensionX=x*factor
nuevaDimensionY=y*factor
dimensiones = np.array([x, y,nuevaDimensionX,nuevaDimensionY,factor]).astype(np.int32)
#matriz de tranformar Xd
matriz=np.zeros([nuevaDimensionX,nuevaDimensionY,3],dtype=np.float32)
matriz.fill(0)
print(dimensiones)
#lo pongo en una dimension
imagen=imagen.reshape(x*y*tres)
matriz=matriz.reshape(nuevaDimensionX*nuevaDimensionY*tres)
funcion(cuda.In(imagen),cuda.InOut(matriz),cuda.In(dimensiones), 
    block=(THREADS,THREADS,1), grid=(int((dimensiones[2]+THREADS)//THREADS),int((dimensiones[3]+THREADS)//THREADS),1 ))
matriz=matriz.reshape(nuevaDimensionX,nuevaDimensionY,tres)
misc.imsave('imagenX2.png', matriz)

