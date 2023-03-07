# %% Setup
import scipy.io as sio
import numpy as np
import imageio as imio
import matplotlib.pyplot as plt

im_multi = sio.loadmat("multispectral_day01.mat")['immulti']
im_masks = imio.imread("annotation_day01.png") == 255

im_mask_salami = np.sum(im_masks, axis=2)
im_mask_unknown = im_masks[:, :, 0]
im_mask_fat = im_masks[:, :, 1]
im_mask_meat = im_masks[:, :, 2]

multi_pixels_fat = np.array([im_multi[im_mask_fat, idx] for idx in range(im_multi.shape[2])])
multi_pixels_meat = np.array([im_multi[im_mask_meat, idx] for idx in range(im_multi.shape[2])])

means_fat = np.mean(multi_pixels_fat, axis=1)
means_meat = np.mean(multi_pixels_meat, axis=1)

stds_fat = np.std(multi_pixels_fat, axis=1)
stds_meat = np.std(multi_pixels_meat, axis=1)

cov_fat = np.cov(multi_pixels_fat)
cov_meat = np.cov(multi_pixels_meat)

m_fat = multi_pixels_fat.shape[1]
m_meat = multi_pixels_meat.shape[1]

cov_pooled = ((m_fat - 1) * cov_fat + (m_meat - 1) * cov_meat) / (m_fat + m_meat - 2)
cov_pooled_inv = np.linalg.inv(cov_pooled)

p_fat = 0.5
p_meat = 0.5

def S_fat(x): 
    return x.T @ cov_pooled_inv @ means_fat - means_fat.T @ cov_pooled_inv @ means_fat / 2 + np.log(p_fat)

def S_meat(x): 
    return x.T @ cov_pooled_inv @ means_meat - means_meat.T @ cov_pooled_inv @ means_meat / 2 + np.log(p_meat)

def classify_multi_pixels(multi_pixels):
    idxs_fat = []
    idxs_meat = []

    for i in range(multi_pixels.shape[1]):
        if S_fat(multi_pixels[:,i]) >= S_meat(multi_pixels[:,i]):
            idxs_fat.append(i)
        else:
            idxs_meat.append(i)
    
    return (idxs_fat, idxs_meat)

def error_rate(multi_pixels_fat, multi_pixels_meat):
    error_count = 0

    for i in range(multi_pixels_fat.shape[1]):
        if S_fat(multi_pixels_fat[:,i]) < S_meat(multi_pixels_fat[:,i]):
            error_count += 1

    for i in range(multi_pixels_meat.shape[1]):
        if S_fat(multi_pixels_meat[:,i]) >= S_meat(multi_pixels_meat[:,i]):
            error_count += 1

    error_rate = error_count / (multi_pixels_fat.shape[1] + multi_pixels_meat.shape[1])

    return (error_rate, error_count)

def classify_im_mutli(im_multi, im_mask):
    im = np.zeros((im_multi.shape[0], im_multi.shape[1], 3), dtype=np.uint8)

    for x in range(im_multi.shape[0]):
        for y in range(im_multi.shape[1]):
            if im_mask[x,y]:
                if S_fat(im_multi[x, y, :]) >= S_meat(im_multi[x, y, :]):
                    im[x, y, 1] = 255
                else:
                    im[x, y, 2] = 255
    
    return im

print(error_rate(multi_pixels_fat, multi_pixels_meat))
im = classify_im_mutli(im_multi, im_mask_salami)
plt.imshow(im)
plt.show()