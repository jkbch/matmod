import os
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage import io
from skimage import color

def video_plot(V):
    fig, ax = plt.subplots()
    for i in range(V.shape[-1]):
        ax.clear()
        ax.imshow(V[:,:,i], cmap = "gray")
        plt.pause(0.01)

dir_path = "toyProblem_F22"
img_paths = os.listdir(dir_path)
img_paths.sort()

imgs = []
for img_path in img_paths:
    img = io.imread(os.path.join(dir_path, img_path))
    img_gray = color.rgb2gray(img)
    imgs.append(img_gray)

V = np.stack(imgs, axis=2)

Vx = V[1:,:,:] - V[0:-1,:,:]
Vy = V[:,1:,:] - V[:,0:-1,:]
Vt = V[:,:,1:] - V[:,:,0:-1]

Vx = ndimage.prewitt(V, 0)
Vy = ndimage.prewitt(V, 1)
Vt = ndimage.prewitt(V, 2)

def Gaussian_Gradient_Filter(V, dim, sd, factor=2):
    x = np.arange(-factor*sd, factor*sd + 1)
    G = 1 / (np.sqrt(2 * np.pi * sd**2)) * np.exp(- x**2 / (2 * sd**2))
    dG = -np.sqrt(2) * x * np.exp(-x**2 / 8) / (16 * np.sqrt(np.pi))

    dV = V
    for i in range(3):
        if(i == dim):
            dV = ndimage.convolve1d(dV, dG, i)
        else:
            dV = ndimage.convolve1d(dV, G, i)

    return dV

Vx = Gaussian_Gradient_Filter(V, 0, 2)
Vy = Gaussian_Gradient_Filter(V, 1, 2)
Vt = Gaussian_Gradient_Filter(V, 2, 2)

for i in range(0,255,10):
  for j in range(0,255,10):

    radius = 2
    px, py, pt = i, j, 0



    px_region = slice(px-radius, px+radius+1)
    py_region = slice(py-radius, py+radius+1)


    A = np.transpose(np.vstack((Vx[px_region, py_region, pt].flatten(), Vy[px_region, py_region, pt].flatten())))
    b = -Vt[px_region, py_region, pt].flatten()

    (x, y), _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    plt.quiver(px,py,x,y,color = "r", scale = 20)


plt.imshow(V[:,:,0],cmap = 'gray')
plt.show()
