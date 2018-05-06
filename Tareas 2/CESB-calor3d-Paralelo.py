import numpy as np
import matplotlib.pyplot as plt
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import matplotlib.animation as animation
    
dx = dy = dz = 0.1
D = 4.
Tcool, Thot = 300, 700
nx, ny, nz = 128, 128, 128
threads_per_block = 8
dx2, dy2, dz2 = dx*dx, dy*dy, dz*dz
dt = dx2 * dy2 * dz2 / (3 * D * (dx2 + dy2 + dz2))

mod = SourceModule("""
    const int nx = 128;
    const int ny = 128;
    const int nz = 128;
    const float D = 4.0;
    const float dx2 = 0.01;
    const float dy2 = 0.01;
    const float dz2 = 0.01; 
    const float dt = dx2 * dy2 * dz2/ (3 * D * (dx2 + dy2 + dz2));
    __global__ void kernelCalor3D(float* u, float* ut){
        int i = threadIdx.x + blockIdx.x*blockDim.x;
        int j = threadIdx.y + blockIdx.y*blockDim.y;
        int k = threadIdx.z + blockIdx.z*blockDim.z;
        
        if(i*(nx-i-1)*j*(ny-j-1)*k*(nz-k-1)!=0){
            ut[k+nz*(j+ny*i)] = u[k+nz*(j+ny*i)] + (D * dt/dx2) * (u[k+nz*(j+(i+1)*ny)] 
                + u[k+nz*(j+(i-1)*ny)] + u[k+nz*((j-1)+i*ny)] + u[k+nz*((j+1)+i*ny)]
                + u[(k+1)+nz*(j+i*ny)] + u[(k-1)+nz*(j+i*ny)] - 6.*u[k+nz*(j+i*ny)]);

        }
    }
""")
# obtener el kernel
kernelCalor3D = mod.get_function("kernelCalor3D")
def animate(data, im):
    im.set_data(data)

def step(u0, u, nsteps):
    for _ in range(nsteps):
        kernelCalor3D(cuda.In(u0), cuda.Out(u), block=(threads_per_block, threads_per_block, threads_per_block), grid=(nx//threads_per_block, ny//threads_per_block, nz//threads_per_block) )
        u0 = u.copy()
        yield get_plane(u, z=32)
#funcion del raul para un plano
def get_plane(u, x=0, y=0, z=0, fix='z'):
    return ([ [u[z+nz*(j+i*ny)] for j in range(ny) ] for i in range(nx) ] if fix is 'z'
        else [ [u[k+nz*(y+i*ny)] for k in range(nz) ] for i in range(nx) ] if fix is 'y'
        else [ [u[k+nz*(j+x*ny)] for k in range(nz) ] for j in range(ny) ])


r, cx, cy, cz = 3, 7, 7, 7 
r2 = r**2
nsteps=200
a0 = [ (Thot if ((i*dx-cx)**2 + (j*dy-cy)**2 + (k*dz-cz)**2) < r2 or i==j or nx-j==i or i%4==0 else Tcool) for i in range(nx) for j in range(ny) for k in range(nz) ]
a = [ 0 for i in range(nx) for j in range(ny) for k in range(nz) ]

u0 = np.array(a0).astype(np.float32)
u = np.array(a).astype(np.float32)
#config
fig, ax = plt.subplots()
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
cbar_ax.set_xlabel('$T$ / K', labelpad=20)
im = ax.imshow(get_plane(u, z=32), cmap=plt.get_cmap('hot'), vmin=Tcool,vmax=Thot)
fig.colorbar(im, cax=cbar_ax)
ax.set_axis_off()
ax.set_title("Mapa de Calor")
#anmacion
ani = animation.FuncAnimation( fig, animate, step(u0, u, nsteps), 
    interval=1, save_count=nsteps, repeat=True,repeat_delay=1, fargs=(im,))
ani.save('animation3d.mp4', fps=20, writer="ffmpeg", codec="libx264")
