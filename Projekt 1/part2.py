import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage import io
from skimage import color


def plot_V(V):
    fig, ax = plt.subplots()
    for i in range(V.shape[-1]):
        ax.clear()
        ax.imshow(V[:,:,i], cmap = "gray")
        plt.pause(0.01)


def create_V(dir_path):
    img_paths = os.listdir(dir_path)
    img_paths.sort()

    imgs = []
    for img_path in img_paths:
        img = io.imread(os.path.join(dir_path, img_path))
        img_gray = color.rgb2gray(img)
        imgs.append(img_gray)

    V = np.stack(imgs, axis=2)

    return V


def gaussian_gradient_filter(V, dim, sd, factor=2):
    x = np.arange(-factor*sd, factor*sd + 1)
    G = 1 / (np.sqrt(2 * np.pi * sd**2)) * np.exp(- x**2 / (2 * sd**2))
    dG = -np.sqrt(2) * x * np.exp(-x**2 / (2 * sd**2))/(2 * np.sqrt(np.pi * sd**2) * sd**2)

    dV = V
    for i in range(3):
        if(i == dim):
            dV = ndimage.convolve1d(dV, dG, i)
        else:
            dV = ndimage.convolve1d(dV, G, i)

    return dV


def diff_V(V):
    #Vx = V[1:,:,:] - V[0:-1,:,:]
    #Vy = V[:,1:,:] - V[:,0:-1,:]
    #Vt = V[:,:,1:] - V[:,:,0:-1]

    #Vx = ndimage.prewitt(V, 0)
    #Vy = ndimage.prewitt(V, 1)
    #Vt = ndimage.prewitt(V, 2)

    sd = 2
    Vx = gaussian_gradient_filter(V, 0, sd)
    Vy = gaussian_gradient_filter(V, 1, sd)
    Vt = gaussian_gradient_filter(V, 2, sd)

    return (Vx, Vy, Vt)


def displacement_pixel(Vx, Vy, Vt, px, py, pt, radius):
    px_region = slice(px-radius, px+radius+1)
    py_region = slice(py-radius, py+radius+1)

    A = np.transpose(np.vstack((Vx[px_region, py_region, pt].flatten(), Vy[px_region, py_region, pt].flatten())))
    b = -Vt[px_region, py_region, pt].flatten()

    x, y = np.linalg.lstsq(A, b, rcond=None)[0]

    return (x, y)


def displacement_frame(Vx, Vy, Vt, V_shape, pt, radius):
    N = 2*radius + 1

    xs = []
    ys = []
    pxs = []
    pys = []

    for px in range(radius+1, V_shape[0], N):
        for py in range(radius+1, V_shape[1], N):
            x, y = displacement_pixel(Vx, Vy, Vt, px, py, pt, radius)

            xs.append(x)
            ys.append(y)
            pxs.append(px)
            pys.append(py)

    return (pxs, pys, xs, ys)


def plot_V_wtih_optical_flow(V, radius):
    N = 2*radius + 1

    Vx, Vy, Vt = diff_V(V)

    fig, ax = plt.subplots()
    for pt in range(0, V.shape[2]):
        pxs, pys, xs, ys = displacement_frame(Vx, Vy, Vt, V.shape, pt, radius)

        ax.clear()
        ax.imshow(V[:,:,pt], cmap = "gray")
        ax.quiver(pys, pxs, ys, xs)
        plt.pause(0.01)

V = create_V("toyProblem_F22")
radius = 4
plot_V_wtih_optical_flow(V, radius)

