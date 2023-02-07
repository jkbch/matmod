import os
import matplotlib.pyplot as plt
import numpy as np
import scipy
from skimage import io
from skimage import color

def plot(V):
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

V = np.stack(imgs,axis=2)

V_x = V[1:,:,:] - V[0:-1,:,:]
V_y = V[:,1:,:] - V[:,0:-1,:]
V_t = V[:,:,1:] - V[:,:,0:-1]

V_x = scipy.ndimage.prewitt(V, 0)
V_y = scipy.ndimage.prewitt(V, 1)
V_t = scipy.ndimage.prewitt(V, 2)

plot(V_t)
