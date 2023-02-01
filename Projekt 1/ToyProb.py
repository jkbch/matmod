#packs
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import ndimage

from skimage import img_as_ubyte
from skimage import img_as_float
from skimage.color import rgb2gray
from os import listdir


#Indlæs 3darray med billeder
dir = "toyProblem_F22/"
img_list=[]

for images in os.listdir(dir):
    img2d = io.imread(dir + images)
    img2d_gray = rgb2gray(img2d)
    img_list.append(img2d_gray)

img3d = np.stack(img_list,axis=2)

#Gradiants
dx=img3d[1:,:,:]-img3d[0:-1,:,:]
dy=img3d[:,1:,:]-img3d[:,0:-1,:]
dt=img3d[:,:,1:]-img3d[:,:,0:-1]

#Gradiants with filter
dx2=ndimage.prewitt(img3d[:,:,:], axis=0)


### plot
fig, ax = plt.subplots()

##### Original
for  i in range(64):
    ax.clear()
    ax.imshow(img3d[:,:,i],cmap = "gray")
    # Note that using time.sleep does *not* work here!
    plt.pause(0.01)

##### Gradient dx
for i in range(64):
    ax.clear()
    ax.imshow(dx[:,:,i],cmap="gray")
    plt.pause(0.01)