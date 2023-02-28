# %%
import numpy as np
from scipy import io

mat = io.loadmat('multispectral_day01.mat')
img = mat['immulti']

print(img.shape)

# %%
