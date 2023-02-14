import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage import io
from skimage import color


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


def plot_V(V):
    fig, ax = plt.subplots()
    for i in range(V.shape[-1]):
        ax.clear()
        ax.imshow(V[:,:,i], cmap = "gray")
        plt.pause(0.01)


def gaussian_gradient_filter(V, dim, sd, factor):
    x = np.arange(-factor * sd // 2, factor * sd // 2 + 1)

    G = 1 / (np.sqrt(2 * np.pi * sd**2)) * np.exp(- x**2 / (2 * sd**2))
    dG = -np.sqrt(2) * x * np.exp(-x**2 / (2 * sd**2))/(2 * np.sqrt(np.pi * sd**2) * sd**2)

    dV = V
    for i in range(3):
        if(i == dim):
            dV = ndimage.convolve1d(dV, dG, i)
        else:
            dV = ndimage.convolve1d(dV, G, i)

    return dV


def gradient_V(V):
    #Vx = V[1:,:,:] - V[0:-1,:,:]
    #Vy = V[:,1:,:] - V[:,0:-1,:]
    #Vt = V[:,:,1:] - V[:,:,0:-1]

    #Vx = ndimage.prewitt(V, 0)
    #Vy = ndimage.prewitt(V, 1)
    #Vt = ndimage.prewitt(V, 2)

    sd = 2
    factor = 6
    Vx = gaussian_gradient_filter(V, 0, sd, factor)
    Vy = gaussian_gradient_filter(V, 1, sd, factor)
    Vt = gaussian_gradient_filter(V, 2, sd, factor)

    return (Vx, Vy, Vt)


def lucas_kanade_solution(Vx_region, Vy_region, Vt_region):
    A = np.transpose(np.vstack((Vx_region.flatten(), Vy_region.flatten())))
    b = -Vt_region.flatten()

    x, y = np.linalg.lstsq(A, b, rcond=None)[0]
    return (x, y)


def displacement_vectors_frame(Vx, Vy, Vt, V_shape, pt, radius):
    N = 2*radius + 1

    xs = []
    ys = []
    pxs = []
    pys = []

    for px in range(radius+1, V_shape[0], N):
        for py in range(radius+1, V_shape[1], N):
            p_indicies = (slice(px-radius, px+radius+1), slice(py-radius, py+radius+1), pt)
            x, y = lucas_kanade_solution(Vx[p_indicies], Vy[p_indicies], Vt[p_indicies])

            xs.append(x)
            ys.append(y)
            pxs.append(px)
            pys.append(py)

    return (pxs, pys, xs, ys)


def plot_V_wtih_optical_flow(V, radius):
    N = 2*radius + 1

    Vx, Vy, Vt = gradient_V(V)

    fig, ax = plt.subplots()
    for pt in range(0, V.shape[2]):
        pxs, pys, xs, ys = displacement_vectors_frame(Vx, Vy, Vt, V.shape, pt, radius)

        ax.clear()
        ax.imshow(V[:,:,pt], cmap = "gray")
        ax.quiver(pys, pxs, ys, [-x for x in xs])
        plt.pause(0.01)


V = create_V("dårligsteVideo")
radius = 10
plot_V_wtih_optical_flow(V, radius)

