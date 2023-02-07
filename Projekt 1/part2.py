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

sd = 2
factor = 3
px = np.arange(-factor*sd, factor*sd + 1)

def G(x):
    return 1 / (np.sqrt(2 * np.pi * sd**2)) * np.exp(- x**2 / (2 * sd**2))

def dG(x):
    return -np.sqrt(2) * x * np.exp(-x**2 / 8) / (16 * np.sqrt(np.pi))

Vx = ndimage.convolve1d(ndimage.convolve1d(ndimage.convolve1d(V, dG(px), 0), G(px), 1), G(px), 2)
Vy = ndimage.convolve1d(ndimage.convolve1d(ndimage.convolve1d(V, G(px), 0), dG(px), 1), G(px), 2)
Vt = ndimage.convolve1d(ndimage.convolve1d(ndimage.convolve1d(V, G(px), 0), G(px), 1), dG(px), 2)

px, py, pt = 200, 220, 0
radius = 2

px_region = slice(px-radius, px+radius+1)
py_region = slice(py-radius, py+radius+1)

A = np.transpose(np.vstack((Vx[px_region, py_region, pt].flatten(), Vy[px_region, py_region, pt].flatten())))
b = -Vt[px_region, py_region, pt].flatten()

(x, y), _, _, _ = np.linalg.lstsq(A, b, rcond=None)

print(x, y)

#plot(V_x)
