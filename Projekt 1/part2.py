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

def img_plot(img):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap = "gray")

dir_path = "toyProblem_F22"
img_paths = os.listdir(dir_path)
img_paths.sort()

imgs = []
for img_path in img_paths:
    img = io.imread(os.path.join(dir_path, img_path))
    img_gray = color.rgb2gray(img)
    imgs.append(img_gray)

V = np.stack(imgs, axis=2)

#Vx = V[1:,:,:] - V[0:-1,:,:]
#Vy = V[:,1:,:] - V[:,0:-1,:]
#Vt = V[:,:,1:] - V[:,:,0:-1]

#Vx = ndimage.prewitt(V, 0)
#Vy = ndimage.prewitt(V, 1)
#Vt = ndimage.prewitt(V, 2)

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

#Vx = Gaussian_Gradient_Filter(V, 0, 2)
#Vy = Gaussian_Gradient_Filter(V, 1, 2)
#Vt = Gaussian_Gradient_Filter(V, 2, 2)

radius = 4
N = 2*radius + 1

# px, py, pt = 200, 220, 0
# px_region = slice(px-radius, px+radius+1)
# py_region = slice(py-radius, py+radius+1)
#
# A = np.transpose(np.vstack((Vx[px_region, py_region, pt].flatten(), Vy[px_region, py_region, pt].flatten())))
# b = -Vt[px_region, py_region, pt].flatten()
#
# x, y = np.linalg.lstsq(A, b, rcond=None)[0]

fig, ax = plt.subplots()
for pt in range(0, V.shape[2]):
    xs = []
    ys = []
    pxs = []
    pys = []

    for px in range(radius+1, V.shape[0], N):
        for py in range(radius+1, V.shape[1], N):
            px_region = slice(px-radius, px+radius+1)
            py_region = slice(py-radius, py+radius+1)

            A = np.transpose(np.vstack((Vx[px_region, py_region, pt].flatten(), Vy[px_region, py_region, pt].flatten())))
            b = -Vt[px_region, py_region, pt].flatten()

            x, y = np.linalg.lstsq(A, b, rcond=None)[0]

            xs.append(x)
            ys.append(y)
            pxs.append(px)
            pys.append(py)

    ax.clear()
    ax.imshow(V[:,:,pt], cmap = "gray")
    ax.quiver(pys, pxs, ys, xs)
    plt.pause(0.01)