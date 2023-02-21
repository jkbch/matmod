#%%
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
import random
from scipy import ndimage

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

num = random.randint(0,2)
print(num)
vek = np.zeros(3)
x = (num == 0) * 1 
y = (num == 1) * 1
t = (num == 2) * 1
gaussgrad_img_x = scipy.ndimage.gaussian_filter1d(img3d, sigma = 4, axis = 0, order = x)
gaussgrad_img_xy = scipy.ndimage.gaussian_filter1d(gaussgrad_img_x, sigma = 4, axis = 1, order = y)
gaussgrad_img_xyt = scipy.ndimage.gaussian_filter1d(gaussgrad_img_xy, sigma = 3, axis = 2, order = t)
#plot(img3d)
#plot(gaussgrad_img_xyt)
#%%
V = img3d
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

#%% Gausfiltre
Vx = Gaussian_Gradient_Filter(V, 0, 2)
Vy = Gaussian_Gradient_Filter(V, 1, 2)
Vt = Gaussian_Gradient_Filter(V, 2, 2)

#%% Prøv på 1 pixel
px, py, pt = 50, 110, 0
radius = 2

px_region = slice(px-radius, px+radius+1)
py_region = slice(py-radius, py+radius+1)

A = np.transpose(np.vstack((Vx[px_region, py_region, pt].flatten(), Vy[px_region, py_region, pt].flatten())))
b = -Vt[px_region, py_region, pt].flatten()

(x, y), _, _, _ = np.linalg.lstsq(A, b, rcond=None)

print(x, y)
origin = np.array([px,py])
vector = np.array([x,y])
#plt.imshow(V[:,:,0], cmap = "gray")
#plt.title('Draw a point on an image with matplotlib \n (case 2 with extent)')
#pil = plt.quiver(*origin, x,y, color=['r'], scale=21)
#plt.savefig("Test/Testcase.png")
#plt.show()

#%% Kør et helt billede
for i in range(0,255,10):
  for j in range(0,255,10):

    radius = 2
    px, py, pt = i, j, 0



    px_region = slice(px-radius, px+radius+1)
    py_region = slice(py-radius, py+radius+1)


    A = np.transpose(np.vstack((Vx[px_region, py_region, pt].flatten(), Vy[px_region, py_region, pt].flatten())))
    b = -Vt[px_region, py_region, pt].flatten()

    (x, y), _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    plt.quiver(px,py,x,y,color = "r", scale = 20)

plt.imshow(V[:,:,0],cmap = 'gray')
plt.savefig("Test/Test.png")
plt.show()

# %%
string = "Test/Billede_"
for k in range(64):
    for i in range(0,255,10):
        for j in range(0,255,10):

            radius = 2
            px, py, pt = i, j, k



            px_region = slice(px-radius, px+radius+1)
            py_region = slice(py-radius, py+radius+1)

            A = np.transpose(np.vstack((Vx[px_region, py_region, pt].flatten(), Vy[px_region, py_region, pt].flatten())))
            b = -Vt[px_region, py_region, pt].flatten()
            (x, y), _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            plt.quiver(px,py,x,y,color = "r", scale = 20)
    tmp_string = string + str(k) + ".png"
    plt.imshow(V[:,:,k],cmap = 'gray')
    plt.savefig(tmp_string)
    plt.show()
    
#%%
path = "Test/"
img_list = []

for images in os.listdir(path):
    img2d = io.imread(path + images)
    #img2d_gray = rgb2gray(img2d)
    img_list.append(img2d)

V_med_gradienter = np.stack(img_list,axis=2)
plot(V_med_gradienter)
#%%

img3d_fucked = np.ones(np.shape(img3d))

#for i in range(np.shape(img3d)[0]):
#    for j in range(np.shape(img3d)[1]):
#        for k in range(np.shape(img3d)[2]):
#            num = random.randint(0,10^6)/(10^6)
#            img3d_fucked[i,j,k] *= num*img3d[i,j,k]
#
#plot(img3d_fucked)