#packs
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy

from skimage import img_as_ubyte
from skimage import img_as_float
from skimage.color import rgb2gray
from os import listdir

def plot(V):
    fig, ax = plt.subplots()
    for i in range(V.shape[-1]):
        ax.clear()
        ax.imshow(V[:,:,i], cmap = "gray")
        plt.pause(0.01)

#def gaus_gradiant_3d(V,simga)



#Indlæs 3darray med billeder
path = "toyProblem_F22/"

img_list = []

for images in os.listdir(path):
    img2d = io.imread(path + images)
    img2d_gray = rgb2gray(img2d)
    img_list.append(img2d_gray)

V = np.stack(img_list,axis=2)

## Gradient

Vx = V[1:,:,:] - V[0:-1,:,:]
Vy = V[:,1:,:] - V[:,0:-1,:]
Vt = V[:,:,1:] - V[:,:,0:-1]

## Filters
# Sobel Kernel
sobkern_img = scipy.ndimage.sobel(V)

# Prewit kernel
V_x = scipy.ndimage.prewitt(V, 0)
V_y = scipy.ndimage.prewitt(V, 1)
V_t = scipy.ndimage.prewitt(V, 2)

# Gaussian gradiant filtering

gauss_x = scipy.ndimage.gaussian_filter1d(V,sigma = 4, axis = 0, order = 0)
gauss_xy = scipy.ndimage.gaussian_filter1d(gauss_x,sigma = 4, axis = 1, order = 0)
gauss_xyt = scipy.ndimage.gaussian_filter1d(gauss_xy,sigma = 4, axis = 2, order = 1)
## Plot
#plot(V)
#plot(V_y)
plot(gauss_xyt)