#packs
from skimage import io
import matplotlib.pyplot as plt
import numpy as np

from skimage import img_as_ubyte
from skimage import img_as_float
from skimage.color import rgb2gray


#Indlæs 3darray med billeder
dir = "toyProblem_F22/"
frame = ["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17",
"18","19","20","21","22","23","24","25","26","27","28","29","30","31","32","33","34","35","36",
"37","38","39","40","41","42","43","44","45","46","47","48","49","50","51","52","53","54","55","56","57","58",
"59","60","61","62","63","64"]
img_list = []
for numframe in frame:
    img2d = io.imread(dir + "frame_"+ numframe + ".png")
    img2d_gray = rgb2gray(img2d)
    img2d_gray = img_as_ubyte(img2d_gray)
    img_list.append(img2d_gray)

img3d = np.stack(img_list,axis=2)




### plot
fig, ax = plt.subplots()

for  i in range(64):
    ax.clear()
    ax.imshow(img3d[:,:,i],cmap = "gray")
    # Note that using time.sleep does *not* work here!
    plt.pause(0.01)

