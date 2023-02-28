# %%
import scipy.io as sio
import numpy as np
import imageio as imio
import matplotlib.pyplot as plt

def load_multi(multi_im_path, annotation_im_path):
    # load the multispectral image
    im = sio.loadmat(multi_im_path)
    multi_im = im['immulti']
        
    # make annotation image of zeros
    annotation_im = np.zeros([multi_im.shape[0],multi_im.shape[1],3],dtype=bool)
    
    # read the mask image
    a_im = imio.v2.imread(annotation_im_path)
    
    # put in ones
    for i in range(0,3):
        annotation_im[:,:,i] = (a_im[:,:,i] == 255) 
    
    return (multi_im, annotation_im)

(multi_im, annotation_im) = load_multi("multispectral_day01.mat", "annotation_day01.png")

# print(multi_im.shape)
# print(annotation_im.shape)

green_means = np.array([np.mean(multi_im[annotation_im[:, :, 1], idx]) for idx in range(multi_im.shape[2])])
red_means = np.array([np.mean(multi_im[annotation_im[:, :, 2], idx]) for idx in range(multi_im.shape[2])])

green_sds = np.array([np.std(multi_im[annotation_im[:, :, 1], idx]) for idx in range(multi_im.shape[2])])
red_sds = np.array([np.std(multi_im[annotation_im[:, :, 2], idx]) for idx in range(multi_im.shape[2])])

# print(green_means)
# print(red_means)
# print(green_sds)
# print(red_sds)

def f(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-1/2 * ((x - mu) / sigma)**2)

idx = 0
x = np.linspace(-20, 20, 100) + (green_means[idx] + red_means[idx]) / 2

# plt.plot(x, f(x, green_means[idx], green_sds[idx]), color="green")
# plt.plot(x, f(x, red_means[idx], red_sds[idx]), color="red")
# plt.show() 

# Find intersections of two functions
t = x[np.argwhere(np.diff(np.sign(f(x, green_means[idx], green_sds[idx]) - f(x, red_means[idx], red_sds[idx])))).flatten()[0]]
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
