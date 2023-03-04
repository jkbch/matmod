# %%
import numpy as np
from scipy import io

pic = io.loadmat('multispectral_day01.mat')
picArray = pic['immulti']

#print(picArray)

# %%

def Sigma(A):
    print(A)
    m=len(A)
    mu = np.zeros((1,m))
    sigma = np.zeros((m,m))
    

    # Compute mean of each row
    for i in range(m):
        mu[i] = np.mean(A,i)
        print(mu[i])
    for a in range(m):
        for b in range(m): 

          sigma[a,b] =  1 / (m-1) * np.sum((A[a,:]-mu) * (A[b,:]-mu))


Sigma(picArray[:,:,0])
# %%
