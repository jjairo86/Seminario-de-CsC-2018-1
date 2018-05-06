import numpy
from pylab import plot, show, grid, xlabel, ylabel,savefig
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from brownian import brownian
def main():
    # The Wiener process parameter.
    delta = 2
    # Total time.
    T = 10.0
    # Number of steps.
    N = 1000
    # Time step size
    dt = T/N
    # Number of realizations to generate.
    m = 10000
    # Create an empty array to store the realizations.
    x = numpy.empty((m,N+1))
    # Initial values of x.
    x[:, 0] = 50

    brownian(x[:,0], N, dt, delta, out=x[:,1:])

    
    
    t = numpy.linspace(0.0, N*dt, N+1)
    
    for k in range(m):
        plot(t, x[k])
    xlabel('t', fontsize=16)
    ylabel('x', fontsize=16)
    grid(True)
    #show()
    	
    savefig("lineas.png")
    plt.close()
    cuarto=N//4
    #pinta 4 histogramas en diferentes tiempos (NOTA: hacer bonito luego)
    mu = 50
    sigma =numpy.std(x[:,cuarto])
    plt.subplot(2, 2, 1)
    n, bins, patches =plt.hist(x[:,cuarto],bins='auto',histtype='stepfilled', facecolor='g',normed=1)
    y = mlab.normpdf(bins,mu, sigma)
    l = plt.plot(bins,y, 'r--', linewidth=1)
    plt.title('t/4 mu = 50 sigma = %i' %sigma)
    plt.ylabel('Frecuencia')
    
    
    sigma =numpy.std(x[:,cuarto*2])
    plt.subplot(2, 2, 2)
    n, bins, patches =plt.hist(x[:,cuarto*2],bins='auto',histtype='stepfilled', facecolor='g',normed=1)
    y = mlab.normpdf(bins,mu, sigma)
    l = plt.plot(bins,y, 'r--', linewidth=1)
    plt.title('t/2 mu = 50 sigma = %i' %sigma)
    plt.ylabel('Frecuencia')

    sigma =numpy.std(x[:,cuarto*3])
    plt.subplot(2, 2, 3)
    n, bins, patches =plt.hist(x[:,cuarto*3],bins='auto',histtype='stepfilled', facecolor='g',normed=1)
    y = mlab.normpdf(bins,mu, sigma)
    l = plt.plot(bins,y, 'r--', linewidth=1)
    plt.title('3t/4 mu = 50 sigma = %i' %sigma)
    plt.ylabel('Frecuencia')

    sigma =numpy.std(x[:,-1])
    plt.subplot(2, 2, 4)
    n, bins, patches =plt.hist(x[:,-1],bins='auto',histtype='stepfilled', facecolor='g',normed=1)
    y = mlab.normpdf(bins,mu, sigma)
    l = plt.plot(bins,y, 'r--', linewidth=1)
    plt.title('t mu=50 sigma=%i' %sigma)
    plt.ylabel('Frecuencia')

    plt.tight_layout()
    plt.savefig("Histogramas.png")
    plt.close()
if __name__ == "__main__":
    main()
    
