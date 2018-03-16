from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
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
#funcion que cambia el color a negativo
def toNegativo(matriz, dim):
    i=0
    nueva=matriz.copy()
    while i<dim-3:
        nueva[i]=255-nueva[i]
        nueva[i+1]=255-nueva[i+1]
        nueva[i+2]=255-nueva[i+2]
        i+=3
    return nueva 
#funcion calidez
def toAlgo(matriz, dim):
    i=0
    nueva=matriz.copy()
    while i<dim-3:
        nueva[i]=nueva[i+1]
        nueva[i+1]=nueva[i+2]
        nueva[i+2]=nueva[i]
        i+=3
    return nueva      
#aqui leo la imagen
imagen= misc.imread('block.jpg')
#obtengo los valores de laas dimensiones de la matriz y el tres es los 3 RGB XD
x,y,tres=imagen.shape
#lo pongo en una dimension
imagen=imagen.reshape(x*y*tres)
#aqui usa la funcion para cambiar a grises luminosity
imagenGray=toGrayLuminosity(imagen,x*y*tres)
#cambia a las 3 dimensines anteriores para mostrar la nueva grises
imagenGray=imagenGray.reshape(x,y,tres)
#usa la funcion para hacerla negativa
imagenNegativa=toNegativo(imagen,x*y*tres)
#cambia a las 3 dimensines anteriores para mostrar la nueva negativa
imagenNegativa=imagenNegativa.reshape(x,y,tres)
#usa la funcion para hacerla calidez
imagenAlgo=toAlgo(imagen,x*y*tres)
#cambia a las 3 dimensines anteriores para mostrar la nueva Algo
imagenAlgo=imagenAlgo.reshape(x,y,tres)
#cambia a las 3 dimensines anteriores para mostrar la original
imagen=imagen.reshape(x,y,tres)
#ver antes
plt.imshow(imagen)
plt.show()
#ver despues grises
plt.imshow(imagenGray)
plt.show()
#ver despues negativa
plt.imshow(imagenNegativa)
plt.show()
#ver despues Algo
plt.imshow(imagenAlgo)
plt.show()