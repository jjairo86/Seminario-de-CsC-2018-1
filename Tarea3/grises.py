import numpy as np
from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit
from scipy.misc import imsave, imread

kernel = '''__global__ void filtro(float *in, float *out){
  int m = threadIdx.y + blockIdx.y * blockDim.y;
  int n = threadIdx.x + blockIdx.x * blockDim.x;

  if(n < %(WIDTH)s && m < %(HEIGHT)s){

    int grey_offset = m * %(WIDTH)s + n;
    int rgb_offset  = grey_offset * %(CHANNELS)s;

    float valor = 0.21f * in[rgb_offset + 0] + 0.71f * in[rgb_offset + 1] + 0.07f * in[rgb_offset + 2];

    out[rgb_offset + 0] = valor;
    out[rgb_offset + 1] = valor;
    out[rgb_offset + 2] = valor;

  }
}'''
image = imread('waifu.png').astype(np.float32)
WIDTH, HEIGHT, CHANNELS = image.shape

kernel = kernel % {
    'HEIGHT'  : HEIGHT,
    'WIDTH'   : WIDTH,
    'CHANNELS': CHANNELS
    }
out = np.copy(image).astype(np.float32)

in_gpu  = gpuarray.to_gpu(image)
out_gpu = gpuarray.to_gpu(out)


mod = compiler.SourceModule(kernel)
func = mod.get_function('filtro')

func(in_gpu,out_gpu,block = (32,32,1), grid = (int((WIDTH+32)//32), int((HEIGHT+32)//32), 1))
out = out_gpu.get()
imsave('waifu2.png',out.reshape(WIDTH,HEIGHT,CHANNELS))
