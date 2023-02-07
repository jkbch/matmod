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



#Indlæs 3darray med billeder
path = "toyProblem_F22/"

img_list = []

for images in os.listdir(path):
    img2d = io.imread(path + images)
    img2d_gray = rgb2gray(img2d)
    img_list.append(img2d_gray)

img3d = np.stack(img_list,axis=2)

## Gradient

Vx = img3d[1:,:,:] - img3d[0:-1,:,:]
Vy = img3d[:,1:,:] - img3d[:,0:-1,:]
Vt = img3d[:,:,1:] - img3d[:,:,0:-1]

## Filters
# Sobel Kernel
sobkern_img = scipy.ndimage.sobel(img3d)


fig, ax = plt.subplots()
for  i in range(64):
    ax.clear()
    ax.imshow(Vx[:,:,i],cmap = "gray")
    plt.pause(0.01)