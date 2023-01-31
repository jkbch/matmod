import skimage as si

toy1 = si.io.imread('toyProblem_F22/frame_01.png')

from matplotlib import pyplot as plt
from matplotlib import image as mpimg
 
plt.title("Toy 1")

plt.imshow(toy1)
plt.show()