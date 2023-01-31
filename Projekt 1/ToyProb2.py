#packs
import os
import matplotlib.pyplot as plt
import numpy as np
import skimage as si

dir_path = "toyProblem_F22"
img_paths = os.listdir(dir_path)

img_list = []
for img_path in img_paths:
    img2d = si.io.imread(os.path.join(dir, img_path))
    img2d_gray = si.rgb2gray(img2d)
    img2d_gray = si.img_as_ubyte(img2d_gray)
    img_list.append(img2d_gray)

img3d = np.stack(img_list,axis=2)

### plot
fig, ax = plt.subplots()

for  i in range(64):
    ax.clear()
    ax.imshow(img3d[:,:,i],cmap = "gray")
    # Note that using time.sleep does *not* work here!
    plt.pause(0.01)

