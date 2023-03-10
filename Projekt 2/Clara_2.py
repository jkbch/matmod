import scipy.io as sio
import numpy as np
import imageio as imio
import matplotlib.pyplot as plt
from numpy import linalg

#loading images
multi_im = sio.loadmat("multispectral_day01.mat")['immulti']
annotation_im = imio.v3.imread("annotation_day01.png") == 255

#The values of the pixels classified as fat/meat in two matrices with shape (19,number of classified pixels).
fat_multi_pixels = np.array([multi_im[annotation_im[:, :, 1], idx] for idx in range(multi_im.shape[2])])
meat_multi_pixels = np.array([multi_im[annotation_im[:, :, 2], idx] for idx in range(multi_im.shape[2])])

#The mean of fat and meat for each spectrum
fat_means = np.mean(fat_multi_pixels, axis=1)
meat_means = np.mean(meat_multi_pixels, axis=1)

#The standard deviation of fat and meat for each spectrum 
fat_stds = np.std(fat_multi_pixels, axis=1)
meat_stds = np.std(meat_multi_pixels, axis=1)

# %%
sigma_fat=np.zeros((19,19))
sigma_meat=np.zeros((19,19))

# Number of pixels classified as fat/meat.
m=fat_multi_pixels.shape[1]
n=meat_multi_pixels.shape[1]

#Computing the covariance matrix 
for a in range(19):
    for b in range(19):
       sigma_fat[a,b]=sum((fat_multi_pixels[a,:]-fat_means[a])*(fat_multi_pixels[b,:]-fat_means[b]))/(m-1) 
       sigma_meat[a,b]=sum((meat_multi_pixels[a,:]-meat_means[a])*(meat_multi_pixels[b,:]-meat_means[b]))/(n-1) 



sigma_pooled=((m-1)*sigma_fat+(n-1)*sigma_meat)/(m+n-2)
sigma_pooled_inv=linalg.inv(sigma_pooled)

# The discriminant function. x is a vector, with the value of a pixel for each spectrum, and mu is a vector of means for each spectrum.
def S(x,mu):
    mult_1=np.matmul(np.transpose(x),np.matmul(sigma_pooled_inv,mu))
    mult_2=np.matmul(np.transpose(mu),np.matmul(sigma_pooled_inv,mu))
    return mult_1-mult_2/2+np.log(1/2)


#Classify function. A is a matrix and rgb is the index for the layer in an rgb image. For red: rgb=0. For green: rgb=1. For blue: rgb=2
def classify(A):
    n=A.shape[0]
    C=np.zeros((n,n))


    for a in range(n):
        for b in range(n):
            if(annotation_im[a,b,:].any()):
                if S(multi_im[a,b,:],fat_means)>=S(multi_im[a,b,:],meat_means):
                    C[a,b]=200
                else:
                    C[a,b]=100
    return C

C=classify(multi_im)

plt.imshow(C)
plt.show()

#C_fat=C_fat.flatten()
#count_fat=np.count_nonzero(C_fat==200)
#count_meat=np.count_nonzero(C_fat==100)

#print(count_fat)
#print(count_meat)


