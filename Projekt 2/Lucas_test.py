#import matplotlib as plt
import matplotlib.pyplot as plt
import numpy as np
import math

mu1 = 175.5
mu2 = 162.9
x = np.linspace(140,200,1000)
f1 = 1 / (6.7*math.sqrt(2*math.pi))*np.exp(-1/2*1/(6.7**2)*(x-mu1)**2)
f2 = 1 / (6.7*math.sqrt(2*math.pi))*np.exp(-1/2*1/(6.7**2)*(x-mu2)**2)


plt.plot(x,f1[:],color='green',linewidth=2)
plt.plot(x,f2[:],color = 'blue', linewidth = 2)
idx = np.argwhere(np.diff(np.sign(f1 - f2))).flatten()

plt.plot(x[idx], f1[idx], 'ro')
plt.show()

# Opgave 4

def multivariate(x,y):
 return 1/(2*math.pi * 2 * 3)* math.exp(-1/2 * (1/4*x**2 + 1/9 * (y-1)**2))


x = np.arange(-5.0,5.0,0.1)
y = np.arange(-5.0,5.0,0.1)
X,Y = np.meshgrid(x, y) # grid of point
Z = 1/(2*math.pi * 2 * 3)* np.exp(-1/2 * (1/4*X**2 + 1/9 * (Y-1)**2))


plt.contourf(X,Y,Z)
plt.title('No correlation')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.show()

rho = 2/3
const = 1 / (12*math.pi  * math.sqrt(1 - rho**2))
const2 = -1/(2 * (1-rho**2))
led1 = 1/4*X**2
T = np.stack([X,Y], axis = 2)
led2 = X*(Y-1)
led3 = 1/9 * (Y - 1)**2
Z_cor = const * np.exp(const2 * (led1 - 2/6*rho*led2 + led3))
#Z_cor =  * math.exp(-1/2 * 1 / (1-rho**2) * ((1/4*X**2 - 3 * rho * np.prod(np.array([X], [(Y-1)]), axis = 1)) + 1/9 * (Y - 1)**2))
plt.contourf(X,Y,Z_cor)
plt.title('Correlation')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.show()


#%%
#%% Lave nogle billeder af de der fucking salamier
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
import random
from scipy import ndimage
import scipy.io

from skimage import img_as_ubyte
from skimage import img_as_float
from skimage.color import rgb2gray
from os import listdir

pic = scipy.io.loadmat('multispectral_day01.mat')
picArray = pic['immulti']
count = -1
for i in range(5):
    for j in range(4):
        count += 1
        if(count > 18):
            tmp = 1      
        else:
            plt.subplot(5,4, count+1)
            plt.imshow(picArray[:,:,int(count)], cmap='gray', vmin=0, vmax=255)
            plt.axis('off')

plt.subplots_adjust(left=0,
                    bottom=0.1,
                    right=0.5,
                    top=0.9,
                    wspace=0.2,
                    hspace=0.4)
plt.show()
# %%
