import pycuda.gpuarray as gpuarray
import pycuda.autoinit
from pycuda.curandom import rand as curand, XORWOWRandomNumberGenerator
import numpy as np
from math import sqrt
from pylab import plot, show, grid, xlabel, ylabel, savefig
import matplotlib.pyplot as plt


def main():
    # The Wiener process parameter.
    delta = 2
    # Total time.
    T = 10.0
    # Number of steps.
    N = 2000
    # Time step size
    dt = T/N
    # Number of realizations to generate.
    m = 1000
    # Create an
    x = np.empty((m,N+1))
    # Initial values of x.
    x[:, 0] = 50
   

    brownian(x[:,0], N, dt, delta, out=x[:,1:])
    t = np.linspace(0.0, N*dt, N+1)
    for k in range(m):
        plot(t, x[k])
    xlabel('t', fontsize=16)
    ylabel('x', fontsize=16)
    grid(True)
    savefig("imagen.png")
    plt.close()


def brownian(x0, n, dt, delta, out=None):
    x0 = np.asarray(x0)
    # For each element of x0, generate a sample of n numbers from a
    # normal distribution.
    r = numGen(size=x0.shape + (n,), desv=delta*sqrt(dt) )
    # if `out` was not given, create an output array.
    if out is None:
        out = np.empty(r.shape)
    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples. 
    np.cumsum(r, axis=-1, out=out)
    out += np.expand_dims(x0, axis=-1)
    return out

def numGen(size, desv):
    i, j = size
    generator = XORWOWRandomNumberGenerator()
    array = generator.gen_normal(shape=i*j, dtype=np.float32) 
    array = array.reshape((i, j)).get()  
    return array







if __name__ == "__main__":
    main()