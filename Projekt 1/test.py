#packs
from skimage import io
import matplotlib.pyplot as plt
import numpy as np

from skimage import img_as_ubyte
from skimage import img_as_float
from skimage.color import rgb2gray

import os
from os import listdir


#Indlæs 3darray med billeder
path = "toyProblem_F22/"

files = os.listdir(path)

img_list = []

print(files)
for images in files:
    img2d = io.imread(path + images)
    img2d_gray = rgb2gray(img2d)
    img2d_gray = img_as_ubyte(img2d_gray)
    img_list.append(img2d_gray)

img3d = np.stack(img_list,axis=2)




### plot
fig, ax = plt.subplots()

for  i in range(64):
    ax.clear()
    ax.imshow(img3d[:,:,i],cmap = "gray")
    # Note that using time.sleep does *not* work here!
    plt.pause(0.01)

