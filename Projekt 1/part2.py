import os
import matplotlib.pyplot as plt
import numpy as np
import skimage as si
from skimage.color import rgb2gray

dir_path = "toyProblem_F22"
img_paths = os.listdir(dir_path)
img_paths.sort()

imgs2d = []
for img_path in img_paths:
    img2d = si.io.imread(os.path.join(dir_path, img_path))
    img2d_gray = rgb2gray(img2d)
    imgs2d.append(img2d_gray)

img3d = np.stack(imgs2d,axis=2)

Vx = img3d[1:,:,:] - img3d[0:-1,:,:]
Vy = img3d[:,1:,:] - img3d[:,0:-1,:]
Vt = img3d[:,:,1:] - img3d[:,:,0:-1]

fig, ax = plt.subplots()
for  i in range(64):
    ax.clear()
    ax.imshow(Vx[:,:,i],cmap = "gray")
    plt.pause(0.01)

