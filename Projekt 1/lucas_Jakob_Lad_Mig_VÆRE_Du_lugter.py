#%%
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
import random

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

#%% Indlæs billeder
path = "toyProblem_F22/"
img_list = []

for images in os.listdir(path):
    img2d = io.imread(path + images)
    img2d_gray = rgb2gray(img2d)
    img_list.append(img2d_gray)

img3d = np.stack(img_list,axis=2)
#plot(img3d)
#%% Gradienter

Vx = img3d[1:,:,:] - img3d[0:-1,:,:]
Vy = img3d[:,1:,:] - img3d[:,0:-1,:]
Vt = img3d[:,:,1:] - img3d[:,:,0:-1]


#%% FILTRE
# Sobel Kernel
sobkern_img = scipy.ndimage.sobel(img3d)


## Plot

#plot(sobkern_img)
# %%
num = random.randint(0,3)
vek = np.zeros(3)
vek[num] = 1
gaussgrad_img_x = scipy.ndimage.gaussian_filter1d(img3d, sigma = 4, axis = 0, order = vek[0])
gaussgrad_img_xy = scipy.ndimage.gaussian_filter1d(gaussgrad_img_x, sigma = 4, axis = 1, order = vek[1])
gaussgrad_img_xyt = scipy.ndimage.gaussian_filter1d(gaussgrad_img_xy, sigma = 3, axis = 2, order = vek[2])
#plot(img3d)
plot(gaussgrad_img_xyt)
# %%
img3d_fucked = np.ones(np.shape(img3d))

for i in range(np.shape(img3d)[0]):
    for j in range(np.shape(img3d)[1]):
        for k in range(np.shape(img3d)[2]):
            num = random.randint(0,10^6)/(10^6)
            img3d_fucked[i,j,k] *= num*img3d[i,j,k]

#plot(img3d_fucked)