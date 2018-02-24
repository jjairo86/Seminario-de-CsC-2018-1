import numpy as np
import matplotlib.pyplot as plt
##PARA ANIMACION
import matplotlib.animation as animation
##

dx = dy = 0.1
D = 4.

Tcool, Thot = 300, 700

nx, ny = 128, 128

dx2, dy2 = dx*dx, dy*dy
dt = dx2 * dy2 / (2 * D * (dx2 + dy2))
def do_timestep(u0, u):
    # Propagate with forward-difference in time, central-difference in space
    for i in range(nx):
        for j in range(ny):
            if i*(nx-i-1)*j*(ny-j-1) != 0:
                u[i][j] = u0[i][j] +  (D * dt/dx2) * ( u0[i-1][j]+u0[i+1][j]+u0[i][j-1]+u0[i][j+1]
                                - 4.0*u0[i][j] )
            
    u0 = u.copy()
    return u0, u

# Initial conditions - ring of inner radius r, width dr centred at (cx,cy) (mm)
r, cx, cy = 4, 5, 5 
r2 = r**2

u0 = np.array([ [ (Thot if ((i*dx-cx)**2 + (j*dy-cy)**2) < r2 else Tcool) for i in range(nx) ] for j in range(ny) ])
u = np.array([ [0 for i in range(nx)] for i in range(ny) ]) 
# Number of timesteps
nsteps = 500
####ESTE QUEDO PARA GRAFICAR
#Esta funcion lo uqe hace es meterle la matriz a la animacion
def animate(data, im):
    im.set_data(data)
#funcion que "genera" los datos
def step(u0,u):
    i=0
    while i<100:
        u0, data = do_timestep(u0, u)
        i+=1
        yield data
#Tdoa la configuracion inicial aqui de la grafica
fig, ax = plt.subplots()
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
cbar_ax.set_xlabel('$T$ / K', labelpad=20)
im = ax.imshow(u, cmap=plt.get_cmap('hot'), vmin=Tcool,vmax=Thot)
fig.colorbar(im, cax=cbar_ax)
ax.set_axis_off()
ax.set_title("Mapa de Calor")
#aqui acaba la configuracion
"""la siguiente linea hace la animacion
recibe la figura, la funcion que actualiza, la funcion que genera,
"""
ani = animation.FuncAnimation(
    fig, animate, step(u0,u), interval=1,save_count=1000, repeat=True,repeat_delay=1, fargs=(im,))
#plt.show()
#ani.save("test.mp4", fps=10)
ani.save('animationMAX.mp4', fps=20, writer="ffmpeg", codec="libx264")
