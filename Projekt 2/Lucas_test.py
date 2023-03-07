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
import math

from skimage import img_as_ubyte
from skimage import img_as_float
from skimage.color import rgb2gray
from os import listdir
from helpFunctions import getPix

def plotMulSpec(day):
    str1 = 'multispectral_day' + day + '.mat'
    str2 = 'color_day' + day + '.png'
    pic = scipy.io.loadmat(str1)
    picArray = pic['immulti']
    count = -1
    for i in range(5):
        for j in range(4):
            count += 1
            if(count > 18):
                colpic = io.imread(str2)
                plt.subplot(5,4, count+1)
                plt.imshow(colpic)
                plt.axis('off')     
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
plotMulSpec('01')

# %%
annotation = io.imread('annotation_day01.png')
pic = scipy.io.loadmat('multispectral_day01.mat')
picArray = pic['immulti']
I, J, K = picArray.shape
plt.imshow(annotation[:,:,2])
plt.colorbar()
plt.show()

def colorArrays(annotation, picArray, color):
    # Returns array of pictures where the annotation pic has the color color
    rgb_array = annotation[:,:,color]
    colorArray = np.zeros(picArray.shape)
    I, J, K = picArray.shape
    for j in range(J):
        for i in range(I):
            if (rgb_array[i,j] > 2):
                colorArray[i,j,:] = picArray[i,j,:]
    return colorArray

def findNonZero(array, layer):
    # Returns a vector of all non zero indexes of a matrix
    I, J, K = array.shape
    non_zero_vec = []
    for j in range(J):
        for i in range(I):
            if array[i,j,layer] != 0:
                non_zero_vec.append(array[i,j,layer])
    return np.array((non_zero_vec))

def meanStdM(colorArray):
    # Returns the means and standarddeviations of all spectral images
    # in an array of pictures
    I, J, K = colorArray.shape
    arr = np.zeros((K,2))
    for k in range(K):
        non_zero = findNonZero(colorArray,k)
        arr[k,0] = np.mean(non_zero)
        arr[k,1] = np.std(non_zero)
    return arr

def normalDistribution(x,mu, std):
    # Returns probabilities of all idices in x from normal
    # distribution with parameters mu and std
    y = 1 / (std * math.sqrt(2 * math.pi)) * np.exp(- (x - mu)**2 / (2*std**2))
    return y

fat_M = colorArrays(annotation, picArray, 1)
meat_M = colorArrays(annotation, picArray, 2)
ms_fat = meanStdM(fat_M)
ms_meat = meanStdM(meat_M)

fat_X = findNonZero(fat_M, 0)
meat_X = findNonZero(meat_M,0)

points = np.linspace(np.min(np.concatenate((fat_X, meat_X))),np.max(np.concatenate((fat_X, meat_X))),200)
fat_Y = normalDistribution(points, ms_fat[0,0], ms_fat[0,1])
meat_Y = normalDistribution(points, ms_meat[0,0], ms_meat[0,1])
plt.plot(points, fat_Y)
plt.plot(points,meat_Y)
plt.show()
idx = np.argwhere(np.diff(np.sign(fat_Y - meat_Y))).flatten()
print(points[idx])
# %%
