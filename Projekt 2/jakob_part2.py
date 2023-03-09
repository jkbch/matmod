# %%
import scipy.io as sio
import numpy as np
import imageio.v2 as imio
import matplotlib.pyplot as plt

days = ["01", "06", "13", "20", "28"]
error_rates = {day:{day:float('nan') for day in days} for day in days}

for train_day in days:
    print("Training day: " + train_day)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
    axs = [ax1, ax2, ax3, ax4]

    for ax, day in zip(axs, [day for day in days if day != train_day]):
        im_multi = sio.loadmat(f'multispectral_day{day}.mat')['immulti']
        im_masks = imio.imread(f'annotation_day{day}.png') == 255

        im_mask_unknown = im_masks[:, :, 0]
        im_mask_fat = im_masks[:, :, 1]
        im_mask_meat = im_masks[:, :, 2]
        im_mask_salami = np.sum(im_masks, axis=2)

        multi_pixels_fat = np.array([im_multi[im_mask_fat, i] for i in range(im_multi.shape[2])])
        multi_pixels_meat = np.array([im_multi[im_mask_meat, i] for i in range(im_multi.shape[2])])

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

        def global_error_rate(multi_pixels_fat, multi_pixels_meat):
            count = 0

            for x in multi_pixels_fat.T:
                if S_fat(x) < S_meat(x):
                    count += 1

            for x in multi_pixels_meat.T:
                if S_fat(x) >= S_meat(x):
                    count += 1

            rate = count / (multi_pixels_fat.shape[1] + multi_pixels_meat.shape[1])

            return (rate, count)

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

        (rate, count) = global_error_rate(multi_pixels_fat, multi_pixels_meat)
        print(f'Day: {day}, error count: {count:02d}, error rate: {rate:.5f}')
        error_rates[train_day][day] = rate

        im = classify_im_mutli(im_multi, im_mask_salami)
        ax.imshow(im)
        ax.set_title(f'Day: {day}')
        ax.set_yticklabels([])
        ax.set_xticklabels([])

    fig.suptitle(f'Train day: {train_day}')
    plt.show()

matrix = [[rate for day, rate in rates.items()] for train_day, rates in error_rates.items()]
print('Error table')
print('\n'.join(['\t'.join([f'{cell:.5f}' for cell in row]) for row in matrix]))

# %%
