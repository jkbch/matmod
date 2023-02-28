# %%
import scipy.io as sio
import numpy as np
import imageio as imio
import matplotlib.pyplot as plt

multi_im = sio.loadmat("multispectral_day01.mat")['immulti']
annotation_im = imio.v3.imread("annotation_day01.png") == 255

print(multi_im.shape)
print(annotation_im.shape)

fat_multi_pixels = np.array([multi_im[annotation_im[:, :, 1], idx] for idx in range(multi_im.shape[2])])
meat_multi_pixels = np.array([multi_im[annotation_im[:, :, 2], idx] for idx in range(multi_im.shape[2])])

print(fat_multi_pixels.shape)
print(meat_multi_pixels.shape)

fat_means = np.mean(fat_multi_pixels, axis=1)
meat_means = np.mean(meat_multi_pixels, axis=1)

fat_stds = np.std(fat_multi_pixels, axis=1)
meat_stds = np.std(meat_multi_pixels, axis=1)

print(fat_means)
print(meat_means)

print(fat_stds)
print(meat_stds)

# %%

def f(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-1/2 * ((x - mu) / sigma)**2)

idx = 0
x = np.linspace(-20, 20, 100) + (fat_means[idx] + meat_means[idx]) / 2

# plt.plot(x, f(x, green_means[idx], green_sds[idx]), color="green")
# plt.plot(x, f(x, red_means[idx], red_sds[idx]), color="red")
# plt.show() 

# Find intersections of two functions
t = x[np.argwhere(np.diff(np.sign(f(x, fat_means[idx], fat_stds[idx]) - f(x, meat_means[idx], meat_stds[idx])))).flatten()[0]]
print(t)

# %%

# x_size = multi_im.shape[0]
# y_size = multi_im.shape[1]
# z_size = multi_im.shape[2]

# mean_greens = np.empty(z_size)
# mean_reds = np.empty(z_size)

# sigma_greens = np.empty(z_size)
# sigma_reds = np.empty(z_size)

# for z in range(z_size):
#     sum_green = 0
#     count_green = 0

#     sum_red = 0
#     count_red = 0

#     for x in range(x_size):
#         for y in range(y_size):
#             if(annotation_im[x, y, 1]):
#                 sum_green += multi_im[x, y, z]
#                 count_green += 1
            
#             if(annotation_im[x, y, 2]):
#                 sum_red += multi_im[x, y, z]
#                 count_red += 1

#     mean_greens[z] = sum_green / count_green
#     mean_reds[z] = sum_red / count_red

#     sigma_sum_green = 0
#     sigma_sum_red = 0

#     for x in range(x_size):
#         for y in range(y_size):
#             if(annotation_im[x, y, 1]):
#                 sigma_sum_green += (multi_im[x, y, z] - mean_greens[z])**2
            
#             if(annotation_im[x, y, 2]):
#                 sigma_sum_red += (multi_im[x, y, z] - mean_reds[z])**2


#     sigma_greens[z] = np.sqrt(1 / (count_green - 1) * sigma_sum_green)
#     sigma_reds[z] = np.sqrt(1 / (count_red - 1) * sigma_sum_red)

# print(mean_greens)
# print(mean_reds)

# print(sigma_greens)
# print(sigma_reds)


# # %%
